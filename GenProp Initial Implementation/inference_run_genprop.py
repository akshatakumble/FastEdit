# ======================
# 0. Setup & Imports
# ======================
!pip install -q numpy opencv-python matplotlib diffusers

import os
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFilter
import random
import matplotlib.pyplot as plt
import numpy as np

from diffusers import StableVideoDiffusionPipeline, DPMSolverMultistepScheduler
from torch.optim import AdamW
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 1. Paths & Config
# ======================

NUM_EPOCHS    = 10
LEARNING_RATE = 5e-5
BATCH_SIZE    = 1
CLIP_LEN      = 8
RESOLUTION    = 256
LATENT_SCALE  = 8
H_LAT         = RESOLUTION // LATENT_SCALE   # 32
W_LAT         = RESOLUTION // LATENT_SCALE   # 32
T_LAT         = CLIP_LEN

FRAMES_DIR = '/content/drive/MyDrive/dataset_davis/DAVIS/JPEGImages/480p'
MASKS_DIR  = '/content/drive/MyDrive/dataset_davis/DAVIS/Annotations/480p'
OUTPUT_DIR = '/content/drive/MyDrive/genprop_object_removal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Name of your trained checkpoint
CKPT_NAME = "genprop_epoch3_run3.pt"  # <-- change if your file is named differently
CKPT_PATH = "/content/drive/MyDrive/genprop_epoch3_run3.pt" #os.path.join(OUTPUT_DIR, CKPT_NAME)

print("Checkpoint path:", CKPT_PATH)

# ======================
# 2. Reload Base SVD Pipeline (fp32)
# ======================

print("Reloading SVD pipeline...")
SVD_CHECKPOINT = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    SVD_CHECKPOINT,
    torch_dtype=torch.float32,
).to(device)

vae   = pipe.vae
unet  = pipe.unet
image_encoder = pipe.image_encoder
sched = pipe.scheduler

# ======================
# 3. Model Components
# ======================

class RegionAwareLoss(nn.Module):
    def __init__(self, lambda_mask=2.0, beta_grad=1.0, gamma_mpd=1.0):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.beta_grad   = beta_grad
        self.gamma_mpd   = gamma_mpd

        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred, target, mask_latent, grad_feats=None,
                mask_pred=None, mask_target=None):

        pred32        = pred.float()
        target32      = target.float()
        mask_latent32 = mask_latent.float()

        # Inside edited region
        inside_pred   = mask_latent32 * pred32
        inside_target = mask_latent32 * target32
        L_mask = self.mse(inside_pred, inside_target)

        # Outside edited region
        outside_mask   = 1.0 - mask_latent32
        outside_pred   = outside_mask * pred32
        outside_target = outside_mask * target32
        L_nonmask = self.mse(outside_pred, outside_target)

        # Gradient term
        if grad_feats is not None:
            grad_feats32 = grad_feats.float()
            grad_masked  = mask_latent32 * grad_feats32
            L_grad = torch.mean(torch.norm(grad_masked, dim=1))
        else:
            L_grad = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        # MPD loss
        if (mask_pred is not None) and (mask_target is not None):
            mpred32   = mask_pred.float()
            mtarget32 = mask_target.float()
            L_mpd = self.bce(mpred32, mtarget32)
        else:
            L_mpd = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        L_total32 = (
            L_nonmask
            + self.lambda_mask * L_mask
            + self.beta_grad * L_grad
            + self.gamma_mpd * L_mpd
        )

        return L_total32, {
            'L_nonmask': L_nonmask.item(),
            'L_mask':    L_mask.item(),
            'L_grad':    L_grad.item(),
            'L_mpd':     L_mpd.item(),
        }


class SelectiveContentEncoder(nn.Module):
    def __init__(self, backbone_unet, num_blocks=14):
        super().__init__()
        self.conv_in = copy.deepcopy(backbone_unet.conv_in)
        self.blocks = nn.ModuleList([
            copy.deepcopy(backbone_unet.down_blocks[i])
            for i in range(num_blocks)
        ])

        self.time_proj = backbone_unet.time_proj
        self.time_embedding = backbone_unet.time_embedding
        self.add_time_proj = backbone_unet.add_time_proj
        self.add_embedding = backbone_unet.add_embedding

    def forward(self, latents_orig, timesteps, added_time_ids,
                image_only_indicator, encoder_hidden_states):

        # Ensure input matches conv_in dtype
        if latents_orig.dtype != self.conv_in.weight.dtype:
            latents_orig = latents_orig.to(self.conv_in.weight.dtype)

        B, C, T, H, W = latents_orig.shape

        # Time embeddings
        t_emb = self.time_proj(timesteps)
        if hasattr(self.time_embedding, 'linear_1'):
            t_emb = t_emb.to(dtype=self.time_embedding.linear_1.weight.dtype)
        emb = self.time_embedding(t_emb)

        aug_emb = self.add_time_proj(added_time_ids.flatten())
        if hasattr(self.add_embedding, 'linear_1'):
            aug_emb = aug_emb.to(dtype=self.add_embedding.linear_1.weight.dtype)
        aug_emb = aug_emb.reshape(B, -1)
        aug_emb = self.add_embedding(aug_emb)

        temb = emb + aug_emb
        temb = temb.repeat_interleave(T, dim=0)

        x = latents_orig.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.conv_in(x)

        if image_only_indicator.ndim == 1:
            image_only_indicator = image_only_indicator.unsqueeze(-1).repeat(1, T)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(T, dim=0)

        feats = []
        for block in self.blocks:
            out = block(
                x,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out

            BT, C_out, H_out, W_out = x.shape
            feat_5d = x.view(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
            feats.append(feat_5d)

        return feats


class MaskPredictionDecoder(nn.Module):
    def __init__(self, in_channels, latent_time, latent_h, latent_w):
        super().__init__()
        self.latent_time = latent_time
        self.latent_h    = latent_h
        self.latent_w    = latent_w

        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, feat):
        x = feat.float()
        x = torch.clamp(x, -10.0, 10.0)

        mask_logits = self.conv(x)   # [B, 1, T', H', W']

        mask_logits = F.interpolate(
            mask_logits,
            size=(self.latent_time, self.latent_h, self.latent_w),
            mode="nearest",
        )
        return mask_logits  # float32


class InjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


import copy

class GenPropModel(nn.Module):
    def __init__(self, backbone, sce, injections, mpd, image_encoder):
        super().__init__()
        self.backbone   = backbone
        self.sce        = sce
        self.injections = nn.ModuleList(injections)
        self.mpd        = mpd
        self.image_encoder = image_encoder

        self._current_injected_feats = None
        self._injection_hooks = []

        # Clear existing hooks
        for block in self.backbone.unet.down_blocks:
            if hasattr(block, "_forward_hooks"):
                block._forward_hooks.clear()

        for idx, block in enumerate(self.backbone.unet.down_blocks[:len(self.injections)]):
            hook = block.register_forward_hook(self._make_injection_hook(idx))
            self._injection_hooks.append(hook)

    def _make_injection_hook(self, idx):
        def hook(module, inputs, output):
            if self._current_injected_feats is None:
                return output

            feat = self._current_injected_feats[idx]    # [B, C, T, H, W]
            B, C, T, H, W = feat.shape
            feat_btchw = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

            if isinstance(output, tuple):
                hidden = output[0]
                feat_btchw = feat_btchw.to(hidden.dtype).to(hidden.device)
                hidden = hidden + feat_btchw
                return (hidden, *output[1:])
            else:
                feat_btchw = feat_btchw.to(output.dtype).to(output.device)
                return output + feat_btchw

        return hook

    def get_image_embeddings(self, images):
        images_resized = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
        dtype = images.dtype
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=images.device, dtype=dtype).view(1, 3, 1, 1)
        images_norm = (images_resized - mean) / std

        image_embeds = self.image_encoder(images_norm).image_embeds
        image_embeds = image_embeds.unsqueeze(1)
        return image_embeds

    def forward(self, V_orig, v1_edited, t, noise, injection_weight: float = 1.0):
        device = V_orig.device
        B, T, C, H, W = V_orig.shape

        dtype = self.backbone.unet.dtype

        # time conditioning
        fps = 6
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        added_time_ids = torch.tensor(
            [fps, motion_bucket_id, noise_aug_strength],
            device=device, dtype=dtype
        )
        added_time_ids = added_time_ids.unsqueeze(0).repeat(B, 1)

        encoder_hidden_states = self.get_image_embeddings(v1_edited)

        # encode original video
        V_flat = V_orig.view(B * T, C, H, W).to(dtype=dtype)
        V_flat = 2.0 * V_flat - 1.0
        lat_flat = self.backbone.vae.encode(V_flat).latent_dist.sample() * 0.18215
        lat_flat = lat_flat.to(dtype)
        _, C_lat, H_lat, W_lat = lat_flat.shape
        latents_orig = lat_flat.view(B, T, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4)

        # encode edited first frame (tiled)
        v1_tiled = v1_edited.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, C, H, W).to(dtype=dtype)
        v1_tiled = 2.0 * v1_tiled - 1.0
        lat_v1_flat = self.backbone.vae.encode(v1_tiled).latent_dist.sample() * 0.18215
        lat_v1_flat = lat_v1_flat.to(dtype)
        latents_v1 = lat_v1_flat.view(B, T, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4)

        # SCE input: concat orig + edited
        x_in = torch.cat([latents_orig, latents_v1], dim=1)
        image_only_indicator = torch.zeros(B, dtype=torch.long, device=device)

        sce_feats = self.sce(
            x_in,
            t,
            added_time_ids,
            image_only_indicator,
            encoder_hidden_states
        )

        # Injection layers
        injected_feats = []
        for inj, feat in zip(self.injections, sce_feats):
            inj_feat = inj(feat)
            if injection_weight != 1.0:
                inj_feat = inj_feat * injection_weight
            injected_feats.append(inj_feat)

        self._current_injected_feats = injected_feats

        # denoising
        noisy_latents = latents_v1 + noise  # (B, 4, T, H, W)

        # Concatenate with latents_orig
        unet_input_latents = torch.cat([noisy_latents, latents_orig], dim=1)  # (B, 8, T, H, W)
        noisy_latents_for_unet = unet_input_latents.permute(0, 2, 1, 3, 4)    # (B, T, 8, H, W)

        unet_out = self.backbone.unet(
            noisy_latents_for_unet,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
        )
        pred_noise = unet_out.sample if hasattr(unet_out, "sample") else unet_out
        pred_noise = pred_noise.view(B, T, C_lat, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)

        self._current_injected_feats = None

        # Mask prediction
        mask_pred = self.mpd(sce_feats[-1])  # float32

        return pred_noise, mask_pred


# ======================
# 4. Dataset
# ======================

class DAVISObjectRemovalDataset(Dataset):
    def __init__(self, frames_dir, masks_dir,
                 clip_len=8,
                 img_transform=None,
                 mask_transform=None):
        self.frames_dir   = Path(frames_dir)
        self.masks_dir    = Path(masks_dir)
        self.clip_len     = clip_len
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

        self.video_names = [p.name for p in self.frames_dir.iterdir() if p.is_dir()]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        vid = self.video_names[idx]
        frame_folder = self.frames_dir / vid
        mask_folder  = self.masks_dir  / vid

        frame_paths = sorted(frame_folder.glob('*.jpg'))
        mask_paths  = sorted(mask_folder.glob('*.png'))

        total = len(frame_paths)
        start = random.randint(0, max(0, total - 1))
        idxs = [(start + i) % total for i in range(self.clip_len)]

        frames = [Image.open(frame_paths[i]).convert('RGB').resize((RESOLUTION, RESOLUTION))
                  for i in idxs]
        masks  = [Image.open(mask_paths[i]).convert('L').resize((RESOLUTION, RESOLUTION))
                  for i in idxs]

        mask1 = masks[0]
        mask1_tensor = T.ToTensor()(mask1)              # [1, H, W]
        mask1_bin    = (mask1_tensor > 0).float()

        frame1_tensor = T.ToTensor()(frames[0])

        blurred = frames[0].filter(ImageFilter.GaussianBlur(radius=15))
        blurred_tensor = T.ToTensor()(blurred)

        mask1_repeat = mask1_bin.repeat(3, 1, 1)  # [3,H,W]
        v1_edited     = mask1_repeat * blurred_tensor + (1 - mask1_repeat) * frame1_tensor

        frames_tensor = torch.stack([T.ToTensor()(f) for f in frames], dim=0)   # [T, C, H, W]

        mask_seq_tensor = torch.stack([(T.ToTensor()(m) > 0).float() for m in masks],
                                      dim=0).unsqueeze(1)                      # [T,1,H,W]

        if self.img_transform is not None:
            frames_tensor = self.img_transform(frames_tensor)
            v1_edited    = self.img_transform(v1_edited)

        if self.mask_transform is not None:
            mask_seq_tensor = self.mask_transform(mask_seq_tensor)

        return frames_tensor, v1_edited, mask_seq_tensor


# ======================
# 5. Build Model & Load Checkpoint
# ======================

channels   = [320, 640]
num_blocks = len(channels)

injection_layers = [
    InjectionLayer(in_channels=channels[i], out_channels=channels[i])
    for i in range(num_blocks)
]

sce = SelectiveContentEncoder(backbone_unet=unet, num_blocks=num_blocks)
mpd = MaskPredictionDecoder(
    in_channels=channels[-1],
    latent_time=T_LAT,
    latent_h=H_LAT,
    latent_w=W_LAT,
)

model = GenPropModel(pipe, sce, injection_layers, mpd, image_encoder).to(device)
print("Model built.")

# ---- Load your trained weights ----
print("Loading checkpoint...")
state_dict = torch.load(CKPT_PATH, map_location=device)

# Optional: align dtypes if needed
model_state = model.state_dict()
for k, v in state_dict.items():
    if k in model_state and isinstance(v, torch.Tensor):
        if v.dtype != model_state[k].dtype:
            state_dict[k] = v.to(model_state[k].dtype)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Checkpoint loaded.")
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)

# Move to mixed precision for inference (optional)
pipe.to(device, dtype=torch.float16)
model.to(device, dtype=torch.float16)
# Ensure MPD runs in float32
model.mpd = model.mpd.to(torch.float32)

model.eval()

# ======================
# 6. Dataloader & Forward Pass
# ======================

dataset    = DAVISObjectRemovalDataset(FRAMES_DIR, MASKS_DIR, clip_len=CLIP_LEN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)

# Take a single batch
V_orig, v1_edited, mask_seq = next(iter(dataloader))
print("V_orig shape:   ", V_orig.shape)    # [1, T, 3, 256, 256]
print("v1_edited shape:", v1_edited.shape) # [1, 3, 256, 256]

V_orig    = V_orig.to(device, dtype=torch.float16)
v1_edited = v1_edited.to(device, dtype=torch.float16)
mask_seq  = mask_seq.to(device, dtype=torch.float16)

# Handle any extra dim like [B,T,1,1,H,W]
if mask_seq.ndim == 6:
    mask_seq = mask_seq.squeeze(2)

B = V_orig.size(0)
timesteps = torch.full(
    (B,),
    sched.config.num_train_timesteps // 2,   # mid-step for consistency
    device=device,
    dtype=torch.long
)

# For visualization we don't really care about pred_noise, so noise can be zeros
noise = torch.zeros(B, 4, T_LAT, H_LAT, W_LAT, device=device, dtype=torch.float16)

with torch.no_grad():
    pred_noise, mask_pred = model(V_orig, v1_edited, timesteps, noise)

print("pred_noise shape:", pred_noise.shape)
print("mask_pred shape: ", mask_pred.shape)   # [1,1,T_lat,H_lat,W_lat]

# 1. Get a sample from the dataset
# We use the same dataset class defined previously

#FramesDir = "/content/drive/MyDrive/dataset_davis/DAVIS_test/JPEGImages/480p"
# FRAMES_DIR FramesDir
dataset = DAVISObjectRemovalDataset(FRAMES_DIR, MASKS_DIR, clip_len=CLIP_LEN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print(dataloader)
# Fetch a batch
V_orig, v1_edited, mask_seq = next(iter(dataloader))

# 2. Prepare the input image for SVD
# v1_edited is [1, 3, H, W], convert to PIL
v1_edited_tensor = v1_edited[0] # [3, H, W]
v1_edited_pil = T.ToPILImage()(v1_edited_tensor)

# ======================
# 7. Decode predicted latents to frames (GENPROP INFERENCE)
# ======================
dataset = DAVISObjectRemovalDataset(FRAMES_DIR, MASKS_DIR, clip_len=CLIP_LEN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
with torch.no_grad():

    # pred_noise is your model's predicted epsilon (shape: [B, 4, T, 32, 32])
    # We must *denoise* and decode it.

    # 1. Start latents = noisy latents used during training
    latents = (v1_edited.to(device) * 2 - 1).to(torch.float16) # Explicitly cast to float16
    latents = pipe.vae.encode(latents).latent_dist.sample() * 0.18215
    latents = latents.unsqueeze(2).repeat(1, 1, T_LAT, 1, 1)   # [B, 4, T, H, W]

    # 2. Perform a single denoising step using your predicted noise
    # (this is not a full sampling loop, but enough to visualize the output)
    latents_denoised = latents - pred_noise

    # 3. Decode each frame using the underlying AutoencoderKL decoder
    decoded_frames = []
    for t in range(T_LAT):
        lat_t = latents_denoised[:, :, t, :, :] / 0.18215 # lat_t is [B, 4, H, W]

        # Create an image_only_indicator for each single frame (batch_size,)
        image_only_indicator = torch.ones(lat_t.shape[0], device=device, dtype=torch.bool)

        # Directly call the AutoencoderKL decoder, which expects a 4D input
        # and now also the image_only_indicator
        frame = pipe.vae.decoder(lat_t, image_only_indicator=image_only_indicator)
        frame = (frame.clamp(-1, 1) + 1) / 2   # back to [0,1]
        decoded_frames.append(frame[0].permute(1, 2, 0).cpu().numpy().astype(np.float32))

# Display results
# ======================
# 8. Show ORIGINAL vs OUTPUT frames
# ======================

# Original frames (convert to numpy and ensure float32)
V_orig_np = V_orig[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)  # [T, H, W, 3]

# Output frames: decoded_frames (already numpy float32)

fig, axs = plt.subplots(2, T_LAT, figsize=(2*T_LAT, 4))

for t in range(T_LAT):
    # ----- Row 1: Original -----
    axs[0, t].imshow(V_orig_np[t])
    axs[0, t].axis("off")
    if t == 0:
        axs[0, t].set_title("Original")

    # ----- Row 2: Output (GenProp) -----
    axs[1, t].imshow(decoded_frames[t])
    axs[1, t].axis("off")
    if t == 0:
        axs[1, t].set_title("Propagated Edit")

plt.tight_layout()
plt.show()