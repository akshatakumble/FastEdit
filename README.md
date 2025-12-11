# Video Inpainting with Region-Aware Propagation

An experimental implementation of Generative Video Propagation (GenProp) for video object removal, built on Stable Video Diffusion (SVD). This repository provides a complete, reproducible framework for training and inference on video inpainting tasks with minimal computational overhead.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Key Components](#key-components)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)

## Overview

EfficientEdit addresses the challenge of video object removal by propagating edits from a single keyframe to an entire video sequence. Unlike traditional frame-by-frame approaches that suffer from temporal inconsistencies, our method leverages the temporal coherence learned by large-scale image-to-video models like Stable Video Diffusion.

### Key Features

- **Parameter-Efficient Design**: Trains only lightweight conditioning modules (~70% reduction in trainable parameters) while keeping the SVD backbone frozen
- **Region-Aware Loss**: Explicitly supervises edited regions while preserving background content
- **Temporal Consistency**: Leverages pre-trained SVD temporal attention for coherent propagation
- **One-Shot Propagation**: Requires only a single edited keyframe and mask to propagate edits across the entire sequence

### Motivation

While GenProp demonstrates impressive results for video editing tasks, the lack of public implementation details presents a barrier to adoption and further research. This repository provides:

1. Experiments and insights into replicating the GenProp architecture using SVD and other parameter-efficient alternatives (Ctrl-Adapter)
2. Detailed analysis of architectural choices and their trade-offs
4. A complete training and inference pipeline

## Architecture

The EfficientEdit framework consists of four main components:

### 1. Frozen SVD Backbone
- **Stable Video Diffusion (SVD)**: Pre-trained image-to-video model providing temporal coherence
- **VAE**: Encodes/decodes frames to/from latent space
- **UNet**: Spatiotemporal denoising network with temporal attention layers
- **Image Encoder**: CLIP-based encoder for conditioning on the edited first frame

### 2. Selective Content Encoder (SCE)
- **Purpose**: Extracts edit-aware features from original and edited video latents
- **Architecture**: ControlNet-style encoder that copies the first N down-blocks of the UNet
- **Input**: Concatenated original video latents and tiled edited first frame latents
- **Output**: Multi-scale features injected into the UNet via adapter layers

### 3. Injection Layers
- **Purpose**: Adapt SCE features to match UNet feature dimensions
- **Architecture**: Lightweight 1×1×1 3D convolutions with GroupNorm
- **Initialization**: Zero-initialized to start with no effect (preserving pre-trained behavior)
- **Integration**: Features are added to corresponding UNet down-blocks via forward hooks

### 4. Mask Prediction Decoder (MPD)
- **Purpose**: Predicts which regions were edited, providing explicit temporal supervision
- **Architecture**: 3D convolutional MLP operating on UNet decoder features
- **Input**: Features from the penultimate up-block of the UNet
- **Output**: Binary mask logits indicating edited regions

### Training Objective

The model is trained with a region-aware loss combining four components:

```
L_total = L_nonmask + λ_mask × L_mask + β_grad × L_grad + γ_mpd × L_mpd
```

- **L_nonmask**: MSE loss on unedited regions (background preservation)
- **L_mask**: MSE loss on edited regions (inpainting quality)
- **L_grad**: Gradient penalty on SCE features within masked regions (boundary sharpness)
- **L_mpd**: Binary cross-entropy loss for mask prediction (temporal supervision)

The model uses v-prediction parameterization, predicting velocity targets rather than noise.

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- PyTorch 2.0+ with CUDA support

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd EfficientEdit-main
```

2. Install dependencies:
```bash
pip install -r GenPropNormal/requirements_runpod.txt
```

Key dependencies include:
- `torch>=2.0.0`
- `diffusers>=0.32.2`
- `transformers>=4.49.0`
- `accelerate>=1.6.0`
- `opencv-python>=4.10.0`
- `wandb` (optional, for logging)

3. Verify installation:
```bash
python -c "import torch; import diffusers; print('Installation successful')"
```

## Dataset Preparation

The training pipeline expects a dataset organized as follows:

```
dataset_root/
├── original_videos/    # Videos with objects to remove
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── edited_videos/      # Ground-truth videos with objects removed
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── masks/              # Binary masks indicating removed regions
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### Dataset Requirements

- **Video Format**: MP4 files with matching filenames across all three directories
- **Resolution**: Videos will be resized to the specified resolution during training (default: 576×576)
- **Frame Count**: Videos should contain at least as many frames as `--num_frames` (default: 8)
- **Mask Format**: Binary masks (0 = background, 1 = object to remove) stored as grayscale videos

### Creating Train/Validation Split

Use the provided script to split your dataset:

```bash
python GenPropNormal/split_validation_data.py \
    --num_samples 50 \
    --train_dir /path/to/train/dataset \
    --val_dir /path/to/val/dataset \
    --seed 42
```

This randomly selects `num_samples` videos from the training set and moves them (along with corresponding edited videos and masks) to the validation directory.

## Training

### Basic Training Command

```bash
python GenPropNormal/train_and_val.py \
    --train_data_dir /path/to/train/dataset \
    --val_data_dir /path/to/val/dataset \
    --output_dir ./checkpoints \
    --val_out_dir ./validation_outputs \
    --epochs 20 \
    --batch_size 1 \
    --num_frames 8 \
    --resolution 576 \
    --lr 5e-5 \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 1 | Batch size (use 1 with gradient accumulation for memory efficiency) |
| `--num_frames` | 8 | Number of frames per video clip |
| `--resolution` | 576 | Input resolution (height and width) |
| `--lr` | 5e-5 | Learning rate |
| `--gradient_accumulation_steps` | 4 | Effective batch size = batch_size × gradient_accumulation_steps |
| `--mixed_precision` | "no" | Use "fp16" or "bf16" for memory efficiency |
| `--gradient_checkpointing` | False | Enable to reduce memory usage (slower training) |
| `--injection_weight` | 1.0 | Weight for SCE feature injection (can be used for ablation) |
| `--val_interval` | 1 | Run validation every N epochs |
| `--save_interval` | 1 | Save checkpoint every N epochs |
| `--wandb_project` | None | Weights & Biases project name (optional) |

### Memory Optimization

For limited GPU memory, use:

```bash
python GenPropNormal/train_and_val.py \
    --train_data_dir /path/to/train \
    --val_data_dir /path/to/val \
    --output_dir ./checkpoints \
    --val_out_dir ./val_outputs \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --resolution 512 \
    --num_frames 6
```

### Training Process

1. **Model Initialization**: 
   - Loads pre-trained SVD from `stabilityai/stable-video-diffusion-img2vid-xt`
   - Initializes SCE by copying UNet down-blocks
   - Creates zero-initialized injection layers and MPD

2. **Forward Pass**:
   - Encodes original video and edited first frame to latents
   - Computes SCE features from concatenated latents
   - Injects features into UNet via forward hooks
   - Predicts noise (velocity) and mask logits

3. **Loss Computation**:
   - Computes region-aware loss with gradient penalty
   - Backpropagates through trainable components only

4. **Checkpointing**:
   - Saves SCE, injection layers, and MPD weights
   - VAE, UNet, and image encoder remain frozen

### Monitoring Training

With Weights & Biases:

```bash
python GenPropNormal/train_and_val.py \
    --wandb_project efficientedit \
    --wandb_entity your-entity \
    ... (other args)
```

Logs include:
- Total loss and component losses (L_mask, L_nonmask, L_grad, L_mpd)
- Training progress and validation metrics
- Validation video outputs

## Inference

### Validation During Training

Validation runs automatically at the specified interval. Outputs are saved to `--val_out_dir/epoch_N/`:
- `val_sample_0.mp4`: Generated video
- `val_sample_0_input.png`: Input edited first frame

### Standalone Inference

To run inference with a trained checkpoint:

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from GenPropNormal.model import GenPropModel, SelectiveContentEncoder, MaskPredictionDecoder, InjectionLayer

# Load checkpoint
checkpoint = torch.load("checkpoint_epoch_17.pt")

# Initialize model (same as training)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float32
)

# Reconstruct model architecture
sce = SelectiveContentEncoder(pipe.unet, num_blocks=2)
injections = [InjectionLayer(320, 320), InjectionLayer(640, 640)]
mpd = MaskPredictionDecoder(in_channels=1280, latent_time=8, latent_h=72, latent_w=72)

model = GenPropModel(pipe, sce, injections, mpd, pipe.image_encoder, pipe.feature_extractor)
model.register_hooks()

# Load weights
model.sce.load_state_dict(checkpoint['sce_state_dict'])
for inj, state in zip(model.injections, checkpoint['injections_state_dict']):
    inj.load_state_dict(state)
model.mpd.load_state_dict(checkpoint['mpd_state_dict'])

# Run inference (see train_and_val.py validate() function for full example)
```

### Inference Parameters

- **num_inference_steps**: Number of diffusion steps (default: 25, higher = better quality)
- **guidance_scale**: Classifier-free guidance scale (default: 3.0)
- **injection_weight**: Weight for SCE features (default: 1.0, can reduce for ablation)


## Key Components

### GenPropModel (`model.py`)

Main model class that orchestrates all components:

- **Forward Pass**: Handles encoding, SCE feature extraction, UNet denoising, and MPD prediction
- **Inference Step**: Single denoising step with classifier-free guidance support
- **Hook Management**: Registers forward hooks for feature injection and MPD feature capture

### SelectiveContentEncoder (`model.py`)

ControlNet-style encoder that processes original and edited latents:

- Copies first N down-blocks from UNet
- Processes concatenated latents: `[latents_orig, latents_v1_cond]`
- Returns multi-scale features for injection

### MaskPredictionDecoder (`model.py`)

3D convolutional decoder predicting edited regions:

- Takes features from penultimate UNet up-block
- Outputs binary mask logits at latent resolution
- Provides explicit supervision for temporal consistency

### RegionAwareLoss (`model.py`)

Combines multiple loss components:

- **MSE on masked regions**: Encourages accurate inpainting
- **MSE on non-masked regions**: Preserves background
- **Gradient penalty**: Sharpens boundaries using finite differences
- **Mask prediction loss**: Supervises MPD output

### GenPropDataset (`dataset.py`)

PyTorch dataset for video loading:

- Loads clips from original, edited, and mask videos
- Applies random temporal sampling
- Returns normalized tensors in `[-1, 1]` range

## Results

### Training Metrics

Our implementation demonstrates consistent learning signals across training:

| Epoch | CLIP Score | LPIPS | SSIM | PSNR (dB) | CLIP-I |
|-------|-----------|-------|------|-----------|--------|
| 1     | 0.78      | 0.51  | 0.08 | 5.12      | -      |
| 10    | 0.81      | 0.48  | 0.11 | 5.90      | -      |
| 17    | **0.82**  | **0.46** | **0.14** | **6.44** | **0.82** |

### Comparison with State-of-the-Art

| Method | CLIP-I Score (↑) |
|--------|------------------|
| ReVideo | 0.9728 |
| SAM + ProPainter | 0.9809 |
| GenProp (Paper) | **0.9879** |
| GenProp (Ours - Epoch 17) | 0.8170 |

*Note: Our model was trained for 17 epochs due to computational constraints. Extended training is expected to close the gap with the fully converged GenProp model.*

### Observations

1. **Semantic Alignment**: CLIP score improved from 0.78 to 0.82, indicating successful learning of semantic editing requirements
2. **Visual Quality**: LPIPS decreased from 0.51 to 0.46, showing improved perceptual quality
3. **Background Preservation**: SSIM increased by ~30% at epoch 17, suggesting the background preservation mechanism was beginning to take effect
4. **Training Efficiency**: The parameter-efficient design enables training on consumer hardware while maintaining generative capabilities

## Limitations and Future Work

### Current Limitations

1. **Training Duration**: Limited to 17 epochs due to GPU constraints, preventing full convergence
2. **Background Preservation**: Low PSNR (6.44 dB) and SSIM (0.14) indicate SCE has not fully learned to preserve unchanged regions
3. **Inference Speed**: Requires 25 diffusion steps, precluding real-time applications
4. **Mask Sensitivity**: Performance depends on mask quality; imprecise segmentations cause boundary artifacts

### Future Directions

1. **Extended Training**: Full SCE fine-tuning to improve background preservation and close the gap with state-of-the-art
2. **Knowledge Distillation**: Reduce inference steps from 25 to 1-4 using our frozen backbone as a teacher model
3. **Mask Refinement**: Integrate learned mask refinement or soft confidence masks for robustness
4. **Text-Guided Inpainting**: Add text conditioning for creative control over filled regions
5. **Multi-Keyframe Conditioning**: Support temporally-varying edits with multiple keyframes
6. **Architectural Improvements**: 
   - Transformer-based SCE for improved long-range modeling
   - Mask-aware gating for Ctrl-Adapter variants
   - Input corruption techniques (e.g., blurring masked regions before encoding)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{efficientedit2024,
  title={EfficientEdit: Video Inpainting with Region-Aware Propagation},
  author={Kumble, Akshata and Gurpur, Amit and Jadia, Raghav and Navindgikar, Rituraj},
  journal={Northeastern University},
  year={2024}
}
```

## Acknowledgments

This implementation is based on the Generative Video Propagation (GenProp) framework proposed by Liu et al. We thank the authors for their foundational work and Stability AI for releasing Stable Video Diffusion.

---

For questions or issues, please open an issue on the repository or contact the authors.
