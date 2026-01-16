import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.preprocessing.normalization.default_normalization_schemes import SpectreNormalization


class SPECTREPlanner(ExperimentPlanner):
    """
    SPECTRE experiment planner for transformer-based segmentation.
    
    This planner is optimized for the SPECTRE transformer architecture with the following
    tested configurations:
    - 40GB GPU: batch size 2, 128x128x128 patches
    - 94GB GPU: batch size 2, 128x320x320 patches
    
    Always uses SPECTRE normalization.
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetSPECTREPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        
        # SPECTRE uses transformer architecture (SEoMT model)
        # We don't set UNet_class here as SPECTRE uses a different model structure
        
        # Reference values based on tested configurations
        # 40GB GPU: batch size 2, 128x128x128 -> ~2,097,152 voxels per sample
        # 94GB GPU: batch size 2, 128x320x320 -> ~13,107,200 voxels per sample
        
        # Using 128x128x128 as baseline for 40GB (conservative estimate)
        self.UNet_reference_val_3d = 850000000  # Estimated for 128^3 patch, batch size 2 on 40GB
        self.UNet_reference_val_2d = 180000000  # Estimated for 2D transformer
        
        self.UNet_reference_val_corresp_GB = 40  # Reference is for 40GB GPU
        self.UNet_reference_val_corresp_bs_2d = 2
        self.UNet_reference_val_corresp_bs_3d = 2
        
        # Transformer architectures prefer cubic/square patches
        self.UNet_featuremap_min_edge_length = 4
        
        # SPECTRE-specific: prefer larger, more uniform patches
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 1024  # Transformers can handle more features
        self.UNet_max_features_3d = 1024
        
        # Transformers work better with larger patches, so we're more conservative
        # with covering the dataset
        self.max_dataset_covered = 0.03  # 3% instead of 5%

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(self) -> Tuple[List[str], List[bool]]:
        """
        SPECTRE always uses SPECTRE normalization scheme.
        """
        if 'channel_names' not in self.dataset_json.keys():
            print("WARNING: 'channel_names' not found in dataset.json. Using 'modality' instead.")
        
        modalities = self.dataset_json['channel_names'] if 'channel_names' in self.dataset_json.keys() else \
            self.dataset_json['modality']
        
        # Always use SPECTRE normalization for all channels
        normalization_schemes = [SpectreNormalization for _ in modalities.values()]
        
        # Mask usage: use mask if the dataset has significant cropping
        if self.dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
            use_mask_for_norm = [True for _ in modalities.values()]
        else:
            use_mask_for_norm = [False for _ in modalities.values()]
        
        normalization_schemes = [i.__name__ for i in normalization_schemes]
        return normalization_schemes, use_mask_for_norm

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        Configurations are unique within each plans file but different plans files can have 
        configurations with the same name. Use SPECTRE-specific identifier.
        """
        return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        """
        Generate plans for SPECTRE transformer architecture.
        Optimized for patch sizes that work well with transformers.
        """
        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())

        # For transformers, we use a simpler architecture specification
        # The actual model (SEoMT) will be instantiated in the trainer
        architecture_kwargs = {
            'network_class_name': 'spectre.models.seomt.SEoMT',
            'arch_kwargs': {
                'num_classes': len(self.dataset_json['labels'].keys()),
                'num_blocks': 4,
                'masked_attn_enabled': True,
                'return_only_final_layer': False,  # For deep supervision
                'upscale_output': True,
                'for_nnunet': True,
                'decoder': False,
            },
            '_kw_requires_import': (),
        }

        # Transformer-specific patch size determination
        # Start with tested configuration as baseline
        tmp = 1 / np.array(spacing)
        
        if len(spacing) == 3:
            # For 3D, start with 128x128x128 as baseline (tested on 40GB)
            # Scale based on spacing to maintain aspect ratio
            initial_patch_size = [round(i) for i in tmp * (128 ** 3 / np.prod(tmp)) ** (1 / 3)]
            
            # Clamp to reasonable transformer sizes (transformers prefer powers of 2)
            initial_patch_size = [min(max(64, p), 320) for p in initial_patch_size]
            
        elif len(spacing) == 2:
            # For 2D, use larger patches (transformers work well with larger contexts)
            initial_patch_size = [round(i) for i in tmp * (512 ** 2 / np.prod(tmp)) ** (1 / 2)]
            initial_patch_size = [min(max(256, p), 768) for p in initial_patch_size]
        else:
            raise RuntimeError("Only 2D and 3D are supported")

        # Clip to median shape
        initial_patch_size = np.minimum(initial_patch_size, median_shape[:len(spacing)])
        
        # Round to multiples of 32 (transformers work better with this)
        patch_size = [int(np.round(p / 32) * 32) for p in initial_patch_size]
        patch_size = [max(64 if len(spacing) == 3 else 256, p) for p in patch_size]  # Minimum sizes
        
        # For transformers, we don't use traditional pooling, but we still need to 
        # define these for compatibility
        if len(spacing) == 3:
            # Assume 4x downsampling in the transformer encoder (patch size 16 or similar)
            num_stages = 5
            pool_op_kernel_sizes = [[1, 1, 1]] + [[2, 2, 2]] * (num_stages - 1)
            conv_kernel_sizes = [[3, 3, 3]] * num_stages
        else:
            num_stages = 6
            pool_op_kernel_sizes = [[1, 1]] + [[2, 2]] * (num_stages - 1)
            conv_kernel_sizes = [[3, 3]] * num_stages
        
        shape_must_be_divisible_by = np.array([2 ** (num_stages - 1)] * len(spacing))
        
        # Ensure patch size is divisible
        patch_size = [int(np.ceil(p / d) * d) for p, d in zip(patch_size, shape_must_be_divisible_by)]

        # VRAM estimation for transformer
        # Transformers have quadratic complexity with respect to sequence length
        # Sequence length = (patch_size[0] * patch_size[1] * patch_size[2]) / (patch_embed_size ** 3)
        # For SPECTRE, patch_embed_size is typically 16, so divide by 4096
        
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)
        
        # Estimate VRAM: this is a rough heuristic for transformers
        # Transformers scale roughly with O(n^2) where n is sequence length
        if len(spacing) == 3:
            baseline_voxels = 128 ** 3  # Our tested baseline
            baseline_vram = self.UNet_reference_val_3d
        else:
            baseline_voxels = 512 ** 2
            baseline_vram = self.UNet_reference_val_2d
        
        current_voxels = np.prod(patch_size)
        # Quadratic scaling for attention, linear for other components (roughly 1.5 power)
        estimate = baseline_vram * (current_voxels / baseline_voxels) ** 1.5
        
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        
        # Adjust patch size if needed
        while estimate > reference and min(patch_size) > (64 if len(spacing) == 3 else 256):
            # Reduce the largest axis
            axis_to_be_reduced = np.argmax(patch_size)
            reduction = 32  # Reduce by 32 at a time
            patch_size[axis_to_be_reduced] = max(patch_size[axis_to_be_reduced] - reduction, 
                                                 64 if len(spacing) == 3 else 256)
            
            # Recalculate estimate
            current_voxels = np.prod(patch_size)
            estimate = baseline_vram * (current_voxels / baseline_voxels) ** 1.5
        
        # Calculate batch size
        batch_size = round((reference / estimate) * ref_bs)
        
        # Cap batch size to cover at most 3% of dataset (conservative for transformers)
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan


if __name__ == '__main__':
    # Test with a dataset
    planner = SPECTREPlanner('Dataset001_BrainTumour', gpu_memory_target_in_gb=40)
    planner.plan_experiment()
