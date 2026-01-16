# SPECTRE - Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers - nnU-Net Integration

## Overview

This repository is a **fork of nnU-Net** that integrates SPECTRE (Self-Supervised & Cross-Modal Pretraining for CT Representation Extraction) functionality. SPECTRE is a self-supervised pretraining method designed to improve medical image segmentation by learning robust feature representations from CT scans.

### State-of-the-art CT Transformer Segmentation Backbone

SPECTRE is a **fully transformer-based** CT foundation model. In the accompanying paper, the SPECTRE-based segmentation setup (SEoMT integrated into nnU-Net) achieves **state-of-the-art performance among transformer-based methods** across multiple established CT semantic segmentation benchmarks, while remaining competitive with strong convolutional baselines.


### Original SPECTRE Implementation

The original SPECTRE code and methodology are available at:
- **Repository**: [https://github.com/cclaess/SPECTRE](https://github.com/cclaess/SPECTRE)
- **Paper**: [https://arxiv.org/abs/2511.17209](https://arxiv.org/abs/2511.17209)

### Compatibility Notice

⚠️ **Important**: This fork follows the same general usage patterns as the official nnU-Net framework, but we **do not guarantee full compatibility** with the standard nnU-Net. There may have been breaking changes introduced during the SPECTRE integration. Please test thoroughly before using in production environments.

## How It Works

This repository finetunes the pretrained SPECTRE encoder weights for medical image segmentation tasks. Depending on the trainer chosen, different decoder architectures will be used with the SPECTRE encoder backbone.

## Spectre Trainers

Current Spectre trainers are implemented and are named:
```
nnUNet_Spectre_L_SEoMT_Trainer (default)
nnUNet_Spectre_L_SEoMT_128_Trainer (used for ablations in paper)
nnUNet_Spectre_L_SEoMT_320_Trainer (reproducing results in paper)
```

### Environment

nnU-Net (and by extension, SPECTRE) requires environment variables to specify where raw data, preprocessed data, and trained models are stored. You must set these three variables before running any nnU-Net commands:

- `nnUNet_raw`: Location of raw datasets
- `nnUNet_preprocessed`: Location for preprocessed data
- `nnUNet_results`: Location for trained models and results

#### Setting Environment Variables

**Linux/MacOS (Permanent)**

Add to your `~/.bashrc` (or `~/.zshrc` for zsh):

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

**Verify Variables**

- Linux/MacOS: `echo $nnUNet_raw`

### Installation and Command Usage

**If nnU-Net is installed** (via `pip install -e .`), you can use the CLI commands directly:

```bash
nnUNetv2_plan_experiment -d <dataset_id> -pl SPECTREPlanner
nnUNetv2_train <dataset_id> <config> <fold> -tr <trainer_name>
```

**If nnU-Net is not installed**, you can call the entry point functions directly from the Python files:

```bash
python nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py -d <dataset_id> -pl SPECTREPlanner
python nnunetv2/run/run_training.py <dataset_id> <config> <fold>
```

The functionality is identical - CLI commands are simply convenient wrappers that are registered during installation.

### Planning and Preprocessing

SPECTRE uses a custom planner (`SPECTREPlanner`) optimized for transformer architectures. To generate plans for your dataset:

```bash
nnUNetv2_plan_experiment -d <dataset_id> -pl SPECTREPlanner -overwrite_plans_name nnUNetSPECTREPlans
```

The SPECTRE planner is optimized based on tested GPU configurations:
- **40GB GPU**: Batch size 2, 128×128×128 patches
- **94GB GPU**: Batch size 2, 128×320×320 patches

The planner automatically:
- Uses **SPECTRE normalization** for all input channels
- Optimizes patch sizes for transformer architecture (prefers powers of 2)
- Accounts for transformer's quadratic attention complexity
- Generates appropriate batch sizes based on available GPU memory

### Training

To run a training with SPECTRE:

```bash
nnUNetv2_train <dataset_id> <config> <fold> -tr nnUNet_Spectre_L_SEoMT_320_Trainer -p nnUNetSPECTREPlans
```

We have exclusively tested the **3d_fullres** config.

### Pretrained Weights

You can specify the path to pretrained SPECTRE checkpoint weights using the `-pretrained_weights` argument:

```bash
nnUNetv2_train <dataset_id> <config> <fold> -tr nnUNet_Spectre_L_SEoMT_320_Trainer -p nnUNetSPECTREPlans -pretrained_weights /path/to/checkpoint.pth
```

- **If a path is provided**: The training will attempt to load the pretrained weights from the specified path.
- **If not provided or not found**: The weights will be automatically downloaded from HuggingFace. We download the DINOv3 + SigLip weights by default.

This allows for flexible weight initialization while ensuring that pretrained weights are always available for training.

## Docker Usage

### Using Pre-built Container

A ready-to-use Docker container is available:

```bash
docker pull sudochris/nnunet_spectre:v2
```

### Building from Dockerfile

Alternatively, you can build the container from the provided Dockerfile:

```bash
docker build -f dockerfile -t nnunet_spectre .
```

## Citation

If you use this code in your research, please cite both SPECTRE and nnU-Net:

### SPECTRE

```bibtex
@misc{claessens2025scalingselfsupervisedcrossmodalpretraining,
      title={Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers}, 
      author={Cris Claessens and Christiaan Viviers and Giacomo D'Amicantonio and Egor Bondarev and Fons van der Sommen},
      year={2025},
      eprint={2511.17209},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.17209}, 
}
```

### nnU-Net

```bibtex
﻿@Article{Isensee2021,
        author={Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
        title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
        journal={Nature Methods},
        year={2021},
        month={Feb},
        day={01},
        volume={18},
        number={2},
        pages={203-211},
        issn={1548-7105},
        doi={10.1038/s41592-020-01008-z},
        url={https://doi.org/10.1038/s41592-020-01008-z}
}

```


### Segmentation Results (from the SPECTRE paper, Table 1)

Average **Dice (%)** on test sets (mean over 5 folds; **KiTS23 uses 4 folds**):

| Method | AMOS-CT | KiTS23 | LiTS | WORD |
|---|---:|---:|---:|---:|
| nnU-Net | 88.87 | 85.99 | 79.29 | 83.11 |
| nnU-Net ResEnc L | – | 88.06 | 81.20 | 85.79 |
| CoTr | 77.13 | 84.63 | 78.44 | 83.11 |
| nnFormer | 85.63 | 75.72 | 77.02 | 82.53 |
| SwinUNETRv2 | 86.37 | 75.07 | 73.27 | 78.99 |
| UNETR | 78.33 | 76.33 | 70.91 | 70.87 |
| WaveFormer | – | 80.61 | – | – |
| Primus-M | – | 86.13 | 79.52 | 83.19 |
| Primus-L | – | 86.04 | 79.90 | 82.99 |
| **SPECTRE (SEoMT)** | **87.55** | **86.64** | **80.14** | **83.31** |

> Source: *Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers* (Table 1). 

### Other changes

TODO: test mirroring


