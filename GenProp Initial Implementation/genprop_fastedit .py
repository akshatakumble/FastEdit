# install required libraries
!pip install -q torch torchvision numpy matplotlib accelerate diffusers wandb

# Cell 1: Setup environment
import os
import gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import random
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from diffusers import StableVideoDiffusionPipeline, DPMSolverMultistepScheduler
from torch.optim import AdamW
from tqdm import tqdm
import copy
import wandb

!wandb login
#os.environ['WANDB_API_KEY'] = '6b8d5ee01f8107768207d7ed33873efcc8b0177d'

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="gurpur-a-northeastern-university",
    # Set the wandb project where this run will be logged.
    project="fastedit",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-4,
        "architecture": "Diffusion",
        "dataset": "DAVIS",
        "epochs": 10,
    },
)

# Mount Google Drive (in Colab)
from google.colab import drive
drive.mount('/content/drive')

# Define dataset paths
FRAMES_DIR = '/content/drive/MyDrive/dataset_davis/DAVIS/JPEGImages/480p'
MASKS_DIR  = '/content/drive/MyDrive/dataset_davis/DAVIS/Annotations/480p'

# Create output / checkpoint directory
OUTPUT_DIR = '/content/drive/MyDrive/genprop_object_removal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Frames dir:", FRAMES_DIR)
print("Masks dir: ", MASKS_DIR)
print("Output dir:", OUTPUT_DIR)

# Basic configuration
CLIP_LEN = 8        # number of consecutive frames
RESOLUTION = 256    # spatial 256×256
BATCH_SIZE = 4      # try 2; adjust if VRAM allows 4
NUM_WORKERS = 2     # data loader workers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# Cell 4: Load pretrained Stable Video Diffusion (SVD) backbone

# Specify checkpoint name
SVD_CHECKPOINT = "stabilityai/stable-video-diffusion-img2vid-xt"  # open-source checkpoint from HuggingFace :contentReference[oaicite:1]{index=1}

# Load pipeline with appropriate dtype
pipe = StableVideoDiffusionPipeline.from_pretrained(
    SVD_CHECKPOINT,
    torch_dtype=torch.float16
)
pipe = pipe.to(device)

# Optionally, enable memory optimizations
pipe.enable_model_cpu_offload()  # offload parts to CPU when not used
pipe.unet.enable_forward_chunking()  # if supported

# Extract components we’ll need
vae   = pipe.vae           # encoder/decoder
unet  = pipe.unet          # U-Net noise predictor
sched = pipe.scheduler      # scheduler for timesteps

# Freeze backbone weights
for param in vae.parameters():
    param.requires_grad = False
for param in unet.parameters():
    param.requires_grad = False

print("Loaded SVD backbone from:", SVD_CHECKPOINT)
print("VAE & U-Net frozen. Device:", device)


# Cell – Full Training Loop (Stable fp32 version, GenProp-style SCE+MPD+hooks)

# --- Memory Cleanup ---
print("Cleaning up GPU memory...")
try:
    del pipe
    del vae
    del unet
    del image_encoder
    del model
    del optimizer
    del loss_module
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()

# --- Configs ---
NUM_EPOCHS    = 10
LEARNING_RATE = 5e-5          # smaller LR for stability
BATCH_SIZE    = 1             # safer for fp32 memory + stability
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

# --- 1. Reload Model in fp32 ---
print("Reloading SVD pipeline...")
SVD_CHECKPOINT = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    SVD_CHECKPOINT,
    torch_dtype=torch.float32
).to(device)

# Extract components
vae   = pipe.vae
unet  = pipe.unet
image_encoder = pipe.image_encoder
sched = pipe.scheduler

# --- 2. Define Classes ---

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

        # Everything is fp32 already, but be explicit
        pred32        = pred.float()
        target32      = target.float()
        mask_latent32 = mask_latent.float()

        # -----------------------------
        # Loss inside edited region
        # -----------------------------
        inside_pred   = mask_latent32 * pred32
        inside_target = mask_latent32 * target32
        L_mask = self.mse(inside_pred, inside_target)

        # -----------------------------
        # Loss outside edited region
        # -----------------------------
        outside_mask   = 1.0 - mask_latent32
        outside_pred   = outside_mask * pred32
        outside_target = outside_mask * target32
        L_nonmask = self.mse(outside_pred, outside_target)

        # -----------------------------
        # Gradient consistency term
        # -----------------------------
        if grad_feats is not None:
            grad_feats32 = grad_feats.float()
            grad_masked  = mask_latent32 * grad_feats32
            L_grad = torch.mean(torch.norm(grad_masked, dim=1))
        else:
            L_grad = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        # -----------------------------
        # MPD loss (mask prediction)
        # -----------------------------
        if (mask_pred is not None) and (mask_target is not None):
            mpred32   = mask_pred.float()
            mtarget32 = mask_target.float()
            L_mpd = self.bce(mpred32, mtarget32)
        else:
            L_mpd = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        # -----------------------------
        # Total loss (fp32)
        # -----------------------------
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

    def forward(self, latents_orig, timesteps, added_time_ids, image_only_indicator, encoder_hidden_states):
        B, C, T, H, W = latents_orig.shape

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        aug_emb = self.add_time_proj(added_time_ids.flatten())
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

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images_norm = (images_resized - mean) / std

        image_embeds = self.image_encoder(images_norm).image_embeds
        image_embeds = image_embeds.unsqueeze(1)
        return image_embeds

    def forward(self, V_orig, v1_edited, t, noise, injection_weight: float = 1.0):
        device = V_orig.device
        B, T, C, H, W = V_orig.shape
        dtype = self.backbone.unet.dtype  # now float32

        # ----- time conditioning -----
        fps = 6
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        added_time_ids = torch.tensor([fps, motion_bucket_id, noise_aug_strength],
                                      device=device, dtype=dtype)
        added_time_ids = added_time_ids.unsqueeze(0).repeat(B, 1)

        encoder_hidden_states = self.get_image_embeddings(v1_edited)

        # ----- encode original video -----
        V_flat = V_orig.view(B * T, C, H, W).to(dtype=dtype)
        V_flat = 2.0 * V_flat - 1.0
        lat_flat = self.backbone.vae.encode(V_flat).latent_dist.sample() * 0.18215
        _, C_lat, H_lat, W_lat = lat_flat.shape
        latents_orig = lat_flat.view(B, T, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4)

        # ----- encode edited first frame (tiled) -----
        v1_tiled = v1_edited.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, C, H, W).to(dtype=dtype)
        v1_tiled = 2.0 * v1_tiled - 1.0
        lat_v1_flat = self.backbone.vae.encode(v1_tiled).latent_dist.sample() * 0.18215
        latents_v1 = lat_v1_flat.view(B, T, C_lat, H_lat, W_lat).permute(0, 2, 1, 3, 4)

        # ----- SCE input: concat orig + edited -----
        x_in = torch.cat([latents_orig, latents_v1], dim=1)
        image_only_indicator = torch.zeros(B, dtype=torch.long, device=device)

        sce_feats = self.sce(
            x_in,
            t,
            added_time_ids,
            image_only_indicator,
            encoder_hidden_states
        )

        # ----- Injection layers -----
        injected_feats = []
        for inj, feat in zip(self.injections, sce_feats):
            inj_feat = inj(feat)   # [B, C_i, T_i, H_i, W_i]
            if injection_weight != 1.0:
                inj_feat = inj_feat * injection_weight
            injected_feats.append(inj_feat)

        self._current_injected_feats = injected_feats

        # ----- denoising -----
        noisy_latents = latents_v1 + noise  # (B, 4, T, H, W)

        # Concatenate with latents_orig to match 8-channel conv_in if needed
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

        # ----- Mask prediction -----
        mask_pred = self.mpd(sce_feats[-1])  # float32

        return pred_noise, mask_pred


# --- Dataset ---

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


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


# --- 3. Setup & Training ---

dataset    = DAVISObjectRemovalDataset(FRAMES_DIR, MASKS_DIR, clip_len=CLIP_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

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

freeze(model.backbone.unet)
freeze(model.backbone.vae)
freeze(model.image_encoder)

unfreeze(model.sce)
for inj in model.injections:
    unfreeze(inj)
unfreeze(model.mpd)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", trainable)
print("Total params:", total)

loss_module = RegionAwareLoss(lambda_mask=2.0, beta_grad=1.0, gamma_mpd=1.0)
optimizer   = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

print(f"Starting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for V_orig, v1_edited, mask_seq in pbar:
        V_orig    = V_orig.to(device)
        v1_edited = v1_edited.to(device)
        mask_seq  = mask_seq.to(device)

        if mask_seq.ndim == 6:
            mask_seq = mask_seq.squeeze(2)

        B = V_orig.size(0)
        timesteps = torch.randint(0, sched.config.num_train_timesteps, (B,), device=device)
        noise = torch.randn(B, 4, T_LAT, H_LAT, W_LAT, device=device)

        pred_noise, mask_pred = model(V_orig, v1_edited, timesteps, noise, injection_weight=1.0)

        # Optional sanity check
        if not torch.isfinite(pred_noise).all():
            print("Non-finite pred_noise detected")
        if not torch.isfinite(mask_pred).all():
            print("Non-finite mask_pred detected")

        if pred_noise.shape[2] != noise.shape[2]:
            pred_noise = F.interpolate(
                pred_noise,
                size=(T_LAT, H_LAT, W_LAT),
                mode='nearest'
            )

        mask_latent = F.interpolate(
            mask_seq.permute(0, 2, 1, 3, 4),
            size=(T_LAT, H_LAT, W_LAT),
            mode='nearest'
        )

        loss, losses_dict = loss_module(pred_noise, noise, mask_latent, None, mask_pred, mask_latent)

        if not torch.isfinite(loss):
            print("Non-finite loss:", losses_dict)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        optimizer.step()

        pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"genprop_epoch{epoch+1}_run_fp32.pt"))

print("Training complete.")