from abc import abstractmethod
from typing import List, Tuple, Union
import torch
from torch import nn, autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_finetune import nnUNetTrainer_finetune
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import spectre.models as spectre_models
from spectre.models import (
    vit_tiny_patch16_128,
    vit_small_patch16_128,
    vit_base_patch16_128,
    vit_large_patch16_128,
)
import wandb


"""
SPECTRE Trainer Module

This module provides trainers for the SPECTRE (SEoMT) architecture with different backbone sizes.

Available Trainers:
- nnUNet_Spectre_L_SEoMT_Trainer: Uses Large ViT backbone

Each trainer automatically loads the appropriate pretrained weights based on the model size.
The backbone registry and weights paths can be customized by modifying the class variables
BACKBONE_REGISTRY and WEIGHTS_REGISTRY in AbstractSpectre.
"""
 
class AbstractSpectre(nnUNetTrainer_finetune):
    # Mapping of model sizes to backbone constructors
    BACKBONE_REGISTRY = {
        "T": vit_tiny_patch16_128,
        "S": vit_small_patch16_128,
        "B": vit_base_patch16_128,
        "L": vit_large_patch16_128,
    }
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-5
        self.llrd: float = 0.8
        self.llrd_l2_enabled: bool = True
        self.lr_mult: float = 1.0
        self.weight_decay = 3e-5
        self.poly_power = 0.9
        self.warmup_steps: List[int] = [500, 1000]
        self.enable_deep_supervision = False
        self.model_size = "L"  # options: "T", "S", "B", "L" - override in subclasses
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 150
    
    def initialize(self):
        """Override to log SPECTRE-specific configuration to wandb after all properties are set."""
        # Call parent initialization first to ensure network is built
        super().initialize()
        
        # Now log SPECTRE-specific hyperparameters to wandb
        # This happens after all child class __init__ methods have run
        if self.local_rank == 0:  # Only log from main process in DDP
            wandb.config.update({
                "model_size": self.model_size,
                "llrd": self.llrd,
                "llrd_l2_enabled": self.llrd_l2_enabled,
                "lr_mult": self.lr_mult,
                "weight_decay": self.weight_decay,
                "poly_power": self.poly_power,
                "warmup_steps": self.warmup_steps,
                "enable_deep_supervision": self.enable_deep_supervision,
                "trainer_class": self.__class__.__name__,
                "lr": self.initial_lr,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
            })
            
            # Log patch size if available
            if hasattr(self.configuration_manager, 'patch_size'):
                wandb.config.update({"patch_size": self.configuration_manager.patch_size})
            
            self.print_to_log_file(f"Logged SPECTRE hyperparameters to wandb: model_size={self.model_size}")

    def get_backbone(self, model_size: str):
        """Get the backbone architecture for the specified model size."""
        if model_size not in self.BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown model size: {model_size}. "
                f"Available sizes: {list(self.BACKBONE_REGISTRY.keys())}"
            )
        
        backbone_fn = self.BACKBONE_REGISTRY[model_size]
        backbone = backbone_fn(
            init_values=1.0,
            pos_embed='rope',
            rope_kwargs={"base": 1000.0}
        )
        
        return backbone

# Concrete trainer implementations for different model sizes

class nnUNet_Spectre_T_SEoMT_Trainer(AbstractSpectre):
    """Trainer using Tiny ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.model_size = "T"

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the SEoMT model with Tiny ViT backbone."""
        backbone = self.get_backbone(self.model_size)
        
        model = spectre_models.SEoMT(
            backbone=backbone,
            num_classes=num_output_channels,
            num_blocks=4,
            masked_attn_enabled=True,
            return_only_final_layer=not enable_deep_supervision,
            for_nnunet=True
        )
        
        return model


class nnUNet_Spectre_S_SEoMT_Trainer(AbstractSpectre):
    """Trainer using Small ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.model_size = "S"

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the SEoMT model with Small ViT backbone."""
        backbone = self.get_backbone(self.model_size)
        
        model = spectre_models.SEoMT(
            backbone=backbone,
            num_classes=num_output_channels,
            num_blocks=4,
            masked_attn_enabled=True,
            return_only_final_layer=not enable_deep_supervision,
            for_nnunet=True
        )
        
        return model


class nnUNet_Spectre_B_SEoMT_Trainer(AbstractSpectre):
    """Trainer using Base ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.model_size = "B"

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the SEoMT model with Base ViT backbone."""
        backbone = self.get_backbone(self.model_size)
        
        model = spectre_models.SEoMT(
            backbone=backbone,
            num_classes=num_output_channels,
            num_blocks=4,
            masked_attn_enabled=True,
            return_only_final_layer=not enable_deep_supervision,
            for_nnunet=True
        )
        
        return model


class nnUNet_Spectre_L_SEoMT_Trainer(AbstractSpectre):
    """Trainer using Large ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.model_size = "L"

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the SEoMT model with Large ViT backbone."""
        backbone = self.get_backbone(self.model_size)
        
        model = spectre_models.SEoMT(
            backbone=backbone,
            num_classes=num_output_channels,
            num_blocks=4,
            masked_attn_enabled=True,
            return_only_final_layer=not enable_deep_supervision,
            for_nnunet=True
        )
        
        return model 
    

class nnUNet_Spectre_L_SEoMT_128_Trainer(nnUNet_Spectre_L_SEoMT_Trainer):
    """Trainer using Large ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (128, 128, 128)  # As per paper
        plans["configurations"][configuration]["batch_size"] = 2
        super().__init__(plans, configuration, fold, dataset_json, device)


    
class nnUNet_Spectre_L_SEoMT_320_Trainer(nnUNet_Spectre_L_SEoMT_Trainer):
    """Trainer using Large ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (128, 320, 320)  # As per paper
        plans["configurations"][configuration]["batch_size"] = 2
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNet_Spectre_L_SEoMT_64_Trainer(nnUNet_Spectre_L_SEoMT_Trainer):
    """Trainer using Large ViT backbone"""
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (64, 64, 64)  # As per paper
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)
       