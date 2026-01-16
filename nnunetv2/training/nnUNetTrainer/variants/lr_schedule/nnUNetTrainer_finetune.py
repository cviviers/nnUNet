from typing import Union, List, Tuple

import torch
from torch._dynamo import OptimizedModule

from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.lr_scheduler.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainer_finetune(nnUNetTrainer):
    """
    Does a finetuning of the entire architecture
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        #### hyperparameters for warmup
        self.initial_lr = 1e-5
        self.llrd: float = 0.8
        self.llrd_l2_enabled: bool = True
        self.lr_mult: float = 1.0
        self.weight_decay = 3e-3
        self.poly_power = 0.9
        self.warmup_steps: List[int] = [500, 1000]
        self.enable_deep_supervision = False
        self.model_size = "L"  # options: "T", "S", "B", "L" - override in subclasses
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 150
        
    def configure_optimizers(self):
        encoder_param_names = {
            n for n, _ in self.network.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.network.backbone.blocks)
        block_i = backbone_blocks

        l2_blocks = torch.arange(
            backbone_blocks - self.network.num_blocks, backbone_blocks
        ).tolist()

        for name, param in reversed(list(self.network.named_parameters())):
            lr = self.initial_lr
            if name.replace("network.backbone.", "") in encoder_param_names:
                name_list = name.split(".")
                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True
                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)

                elif (is_block or block_i == 0) and self.lr_mult != 1.0:
                    lr *= self.lr_mult

                if "backbone.norm" in name:
                    lr = self.initial_lr

                if (
                    is_block
                    and (block_i in l2_blocks)
                    and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0))
                ):
                    lr = self.initial_lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.initial_lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = self.optimizer_class(param_groups, weight_decay=self.weight_decay)
        lr_scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            poly_power=self.poly_power,
        )

        return optimizer, lr_scheduler
    
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        We need to overwrite that entire function because we need to fiddle the correct optimizer in between
        loading the checkpoint and applying the optimizer states. Yuck.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else self.inference_allowed_mirroring_axes
        )

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])