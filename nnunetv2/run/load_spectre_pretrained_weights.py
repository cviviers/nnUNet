import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from pathlib import Path


def download_from_huggingface(model_size: str = "L") -> str:
    """
    Download pretrained weights from HuggingFace if not already cached.
    
    Args:
        model_size: Model size (T, S, B, L)
    
    Returns:
        Path to the downloaded weights file
    """
    from huggingface_hub import hf_hub_download
    
    # Mapping of model sizes to HuggingFace filenames
    HF_FILENAMES = {
        "T": "spectre_backbone_vit_tiny_patch16_128.pt",
        "S": "spectre_backbone_vit_small_patch16_128.pt",
        "B": "spectre_backbone_vit_base_patch16_128.pt",
        "L": "spectre_backbone_vit_large_patch16_128.pt",
    }
    
    if model_size not in HF_FILENAMES:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(HF_FILENAMES.keys())}")
    
    filename = HF_FILENAMES[model_size]
    
    print(f"Downloading SPECTRE {model_size} backbone from HuggingFace...")
    try:
        weights_path = hf_hub_download(
            repo_id="cclaess/SPECTRE",
            filename=filename,
            repo_type="model"
        )
        print(f"Successfully downloaded weights to: {weights_path}")
        return weights_path
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        raise


def load_spectre_pretrained_weights(backbone, fname, verbose=False):
    """
    
    """
    if fname is None:
        # load from huggingface
        raise ValueError("No filename provided for loading pretrained weights.")
    else:
        if dist.is_initialized():
            saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()), weights_only=True)
        else:
            saved_model = torch.load(fname, weights_only=True)

        print(f"Loaded pretrained weights from {fname}")
    print("Verifying compatibility of pretrained weights with current model...")
    # print(saved_model)
    
    if isinstance(backbone, DDP):
        mod = backbone.module
    else:
        mod = backbone
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    #model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    # for key, _ in model_dict.items():
    #     if all([i not in key for i in skip_strings_in_pretrained]):
    #         assert key in pretrained_dict, \
    #             f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
    #             f"compatible with your encoder."
    #         assert model_dict[key].shape == pretrained_dict[key].shape, \
    #             f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
    #             f"{pretrained_dict[key].shape}; your encoder: {model_dict[key]}. The pretrained model " \
    #             f"does not seem to be compatible with your encoder."

    # fun fact: in principle this allows loading from parameters that do not cover the entire encoder. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}
    mod.load_state_dict(saved_model, strict=False)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                    if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    # model_dict.update(pretrained_dict)

    # print("################### Loading pretrained weights from file ", fname, '###################')
    # if verbose:
    #     print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
    #     for key, value in pretrained_dict.items():
    #         print(key, 'shape', value.shape)
    #     print("################### Done ###################")
    # mod.load_state_dict(model_dict)


