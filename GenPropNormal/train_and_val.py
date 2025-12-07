import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers import StableVideoDiffusionPipeline, EulerDiscreteScheduler
from diffusers.utils import export_to_video
from torch.utils.checkpoint import checkpoint
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from dataset import GenPropDataset
from model import GenPropModel, SelectiveContentEncoder, MaskPredictionDecoder, InjectionLayer, RegionAwareLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train GenProp Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--val_out_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fast_validation", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--val_num_frames", type=int, default=None,
                        help="Number of frames for validation. Defaults to --num_frames.")
    parser.add_argument("--injection_weight", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def print_stats(name, tensor):
    if tensor.numel() == 0:
        return
    t = tensor.detach().float()
    print(f"  [Stats] {name}: Shape={tuple(t.shape)} | Min={t.min():.3f} | Max={t.max():.3f} | Mean={t.mean():.3f} | Std={t.std():.3f}")


def save_video(tensor, path, fps=6):
    """Save tensor [B, C, T, H, W] or [C, T, H, W] to video"""
    if tensor.ndim == 5:
        vid_tensor = tensor[0].detach().cpu()
    else:
        vid_tensor = tensor.detach().cpu()
    
    print_stats("Video Output Tensor", vid_tensor)
    vid = ((vid_tensor + 1) / 2).clamp(0, 1) * 255
    vid = vid.permute(1, 2, 3, 0).contiguous().to(torch.uint8).numpy()
    
    T, H, W, C = vid.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        print(f"Failed to open video writer for {path}")
        return

    for i in range(T):
        frame = vid[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print(f"Saved video to {path}")


def tensor_to_pil(tensor):
    """Convert tensor [C, H, W] in [-1, 1] to PIL Image"""
    img = ((tensor + 1) / 2).clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency"""
    if hasattr(model.unet, 'enable_gradient_checkpointing'):
        model.unet.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing for UNet")
    
    if hasattr(model.sce, 'blocks'):
        for block in model.sce.blocks:
            if hasattr(block, 'enable_gradient_checkpointing'):
                block.enable_gradient_checkpointing()
            elif hasattr(block, 'gradient_checkpointing'):
                block.gradient_checkpointing = True
        print("Enabled gradient checkpointing for SCE blocks")


def validate(model, val_loader, scheduler, device, epoch, val_out_dir,
             T_LAT, H_LAT, W_LAT, args, accelerator, fast_mode=False):
    """Validation loop using the GenPropModel's inference_step method."""
    if not accelerator.is_main_process:
        return
        
    print(f"\n{'='*60}")
    print(f"Running Validation for Epoch {epoch} (Fast Mode: {fast_mode})")
    print(f"{'='*60}")
    
    epoch_dir = os.path.join(val_out_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    
    NUM_FRAMES = args.val_num_frames if args.val_num_frames else args.num_frames
    print(f"Using {NUM_FRAMES} frames for validation")
    NUM_STEPS = args.num_inference_steps
    FPS = 7
    MOTION_BUCKET = 127
    NOISE_AUG_STRENGTH = 0.02
    MIN_GUIDANCE = 1.0
    MAX_GUIDANCE = args.guidance_scale
    
    fresh_scheduler = unwrapped_model.backbone.scheduler
    
    with torch.no_grad():
        for i, (V_orig, V_edited, v1_edited, _) in enumerate(val_loader):
            V_orig = V_orig.to(device)
            V_edited = V_edited.to(device)
            v1_edited = v1_edited.to(device)
            
            print(f"\n--- Validation Video {i} ---")
            v1_pil = tensor_to_pil(v1_edited[0])
            
            image = unwrapped_model.backbone.video_processor.preprocess(
                [v1_pil], height=args.resolution, width=args.resolution
            ).to(device, dtype=torch.float32)
            
            seed = 42 + i
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(image.shape, generator=generator, device=device, dtype=image.dtype)
            image = image + NOISE_AUG_STRENGTH * noise
            
            image_latents = unwrapped_model.vae.encode(image).latent_dist.mode()
            cond_latents = image_latents.unsqueeze(1).repeat(1, NUM_FRAMES, 1, 1, 1)
            cond_latents = cond_latents.permute(0, 2, 1, 3, 4)
            print_stats("cond_latents", cond_latents)
            
            clip_image = unwrapped_model.feature_extractor(
                images=v1_pil, return_tensors="pt"
            ).pixel_values.to(device=device, dtype=torch.float32)
            image_embeddings = unwrapped_model.image_encoder(clip_image).image_embeds.unsqueeze(1)
            print_stats("image_embeddings", image_embeddings)
            
            added_time_ids = torch.tensor(
                [[FPS - 1, MOTION_BUCKET, NOISE_AUG_STRENGTH]],
                device=device, dtype=torch.float32
            )
            
            fresh_scheduler.set_timesteps(NUM_STEPS, device=device)
            timesteps = fresh_scheduler.timesteps
            print(f"Timesteps: {timesteps[:5]}... (len={len(timesteps)})")
            
            latent_generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (1, 4, NUM_FRAMES, H_LAT, W_LAT),
                generator=latent_generator, device=device, dtype=torch.float32,
            )
            latents = latents * fresh_scheduler.init_noise_sigma
            print_stats("Initial latents", latents)
            
            print("Denoising...")
            for step_idx, t in enumerate(tqdm(timesteps, desc=f"Val {i}")):
                latent_model_input = fresh_scheduler.scale_model_input(latents, t)
                
                if step_idx == 0:
                    print_stats(f"latent_model_input (step 0)", latent_model_input)
                
                noise_pred = unwrapped_model.inference_step(
                    latents=latent_model_input,
                    cond_latents=cond_latents,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    t=t,
                    do_cfg=True,
                    guidance_scale=MAX_GUIDANCE,
                    V_orig=V_orig if args.injection_weight > 0 else None,
                    v1_edited=v1_edited if args.injection_weight > 0 else None,
                    injection_weight=args.injection_weight
                )
                
                if step_idx == 0:
                    print_stats(f"noise_pred (step 0)", noise_pred)
                
                latents_for_step = latents.permute(0, 2, 1, 3, 4).reshape(-1, 4, H_LAT, W_LAT)
                noise_pred_for_step = noise_pred.permute(0, 2, 1, 3, 4).reshape(-1, 4, H_LAT, W_LAT)
                
                stepped = fresh_scheduler.step(noise_pred_for_step, t, latents_for_step, return_dict=False)[0]
                latents = stepped.reshape(1, NUM_FRAMES, 4, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)
                
                if step_idx == 0:
                    print_stats(f"latents after step 0", latents)
            
            print_stats("Final latents", latents)
            
            latents_decode = latents.permute(0, 2, 1, 3, 4).reshape(-1, 4, H_LAT, W_LAT)
            latents_decode = latents_decode / unwrapped_model.vae.config.scaling_factor
            print_stats("latents for decode", latents_decode)
            
            frames = unwrapped_model.vae.decode(latents_decode, num_frames=NUM_FRAMES).sample
            frames = (frames / 2 + 0.5).clamp(0, 1)
            print_stats("decoded frames", frames)
            
            frames_np = frames.cpu().permute(0, 2, 3, 1).numpy()
            frames_np = (frames_np * 255).round().astype("uint8")
            pil_frames = [Image.fromarray(f) for f in frames_np]
            
            save_path = os.path.join(epoch_dir, f"val_sample_{i}.mp4")
            export_to_video(pil_frames, save_path, fps=FPS)
            print(f"Saved to {save_path}")
            
            input_path = os.path.join(epoch_dir, f"val_sample_{i}_input.png")
            v1_pil.save(input_path)
            print(f"Saved input to {input_path}")


def main():
    args = parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.wandb_project else None,
    )
    
    set_seed(args.seed)
    
    device = accelerator.device
    print(f"Training on {device} with {accelerator.num_processes} process(es)")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    
    if args.wandb_project and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"entity": args.wandb_entity}}
        )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.val_out_dir, exist_ok=True)

    train_dataset = GenPropDataset(args.train_data_dir, resolution=args.resolution,
                                   clip_len=args.num_frames, split='train')
    val_dataset = GenPropDataset(args.val_data_dir, resolution=args.resolution,
                                 clip_len=args.num_frames, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print("Loading SVD backbone...")
    SVD_CHECKPOINT = "stabilityai/stable-video-diffusion-img2vid-xt"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SVD_CHECKPOINT, torch_dtype=torch.float32,
    )
    
    LATENT_SCALE = 8
    H_LAT = args.resolution // LATENT_SCALE
    W_LAT = args.resolution // LATENT_SCALE
    T_LAT = args.num_frames

    channels = [320, 640]
    num_blocks = len(channels)
    
    injection_layers = [
        InjectionLayer(in_channels=channels[i], out_channels=channels[i])
        for i in range(num_blocks)
    ]
    
    sce = SelectiveContentEncoder(backbone_unet=pipe.unet, num_blocks=num_blocks)
    
    num_up_blocks = len(pipe.unet.up_blocks)
    penultimate_up_block = pipe.unet.up_blocks[num_up_blocks - 2]
    
    mpd_in_channels = None
    if hasattr(penultimate_up_block, 'out_channels'):
        mpd_in_channels = penultimate_up_block.out_channels
    elif hasattr(penultimate_up_block, 'resnets') and len(penultimate_up_block.resnets) > 0:
        last_resnet = penultimate_up_block.resnets[-1]
        if hasattr(last_resnet, 'out_channels'):
            mpd_in_channels = last_resnet.out_channels
        elif hasattr(last_resnet, 'conv2'):
            mpd_in_channels = last_resnet.conv2.out_channels
        elif hasattr(last_resnet, 'spatial_res_block') and hasattr(last_resnet.spatial_res_block, 'conv2'):
            mpd_in_channels = last_resnet.spatial_res_block.conv2.out_channels
    
    if mpd_in_channels is None:
        print(f"Penultimate up_block type: {type(penultimate_up_block)}")
        print(f"Penultimate up_block attributes: {[a for a in dir(penultimate_up_block) if not a.startswith('_')]}")
        if hasattr(penultimate_up_block, 'resnets') and len(penultimate_up_block.resnets) > 0:
            print(f"Resnet type: {type(penultimate_up_block.resnets[-1])}")
            print(f"Resnet attributes: {[a for a in dir(penultimate_up_block.resnets[-1]) if not a.startswith('_')]}")
        mpd_in_channels = 1280
        print(f"Using fallback MPD input channels: {mpd_in_channels}")
    else:
        print(f"MPD input channels (from penultimate up_block): {mpd_in_channels}")
    
    mpd = MaskPredictionDecoder(in_channels=mpd_in_channels, latent_time=T_LAT,
                                latent_h=H_LAT, latent_w=W_LAT)
    
    model = GenPropModel(
        pipe, sce, injection_layers, mpd,
        pipe.image_encoder, pipe.feature_extractor
    )
    
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    unet_dtype = pipe.unet.dtype
    for inj in model.injections:
        inj.to(dtype=unet_dtype)
    model.mpd.to(dtype=unet_dtype)
    model.sce.to(dtype=unet_dtype)

    model.backbone.vae.requires_grad_(False)
    model.backbone.unet.requires_grad_(False)
    model.image_encoder.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")
    
    optimizer = AdamW(trainable_params, lr=args.lr)
    loss_module = RegionAwareLoss()
    scheduler = pipe.scheduler

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    loss_module = loss_module.to(device)

    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = args.epochs * num_update_steps_per_epoch
    print(f"Total training steps: {total_training_steps}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        unwrapped = accelerator.unwrap_model(model)
        print(f"DEBUG: _injection_hooks = {len(unwrapped._injection_hooks)}, _mpd_hook = {unwrapped._mpd_hook}")
        print(f"DEBUG: num up_blocks = {len(unwrapped.unet.up_blocks)}")
        
        unwrapped.register_hooks()
        
        print(f"DEBUG after register: _injection_hooks = {len(unwrapped._injection_hooks)}, _mpd_hook = {unwrapped._mpd_hook}")
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                    disable=not accelerator.is_main_process)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for step, (V_orig, V_edited, v1_edited, mask_seq) in enumerate(pbar):
            with accelerator.accumulate(model):
                B = V_orig.shape[0]
                T = V_orig.shape[1]
                noise = torch.randn(B, 4, T_LAT, H_LAT, W_LAT, device=device, dtype=torch.float32)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)

                # Prepare mask_latent
                if mask_seq.ndim == 5:
                    mask_latent = F.interpolate(
                        mask_seq.permute(0, 2, 1, 3, 4),  # [B, 1, T, H, W]
                        size=(T_LAT, H_LAT, W_LAT),
                        mode='nearest'
                    )
                else:
                    mask_latent = mask_seq

                # Forward pass with V_orig (for SCE), V_edited (target for denoising), v1_edited (conditioning)
                # Pass scheduler for v-prediction target computation
                pred_noise, mask_pred, grad_feats, velocity_target = model(
                    V_orig, V_edited, v1_edited, timesteps, noise,
                    scheduler=scheduler,
                    mask_latent=mask_latent,
                    compute_grad_loss=True
                )

                # Use velocity_target instead of noise for v-prediction loss
                loss, losses_dict = loss_module(pred_noise, velocity_target, mask_latent, grad_feats, mask_pred, mask_latent)

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                
            epoch_loss += loss.detach().item()
            num_batches += 1
            
            if accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix(loss=loss.item(), avg_loss=avg_loss)
                
                if args.wandb_project and accelerator.sync_gradients:
                    log_data = {
                        "train/total_loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "epoch": epoch,
                        "global_step": global_step
                    }
                    for k, v in losses_dict.items():
                        log_data[f"train/{k}"] = v
                    accelerator.log(log_data, step=global_step)

        if epoch % args.val_interval == 0:
            accelerator.wait_for_everyone()
            validate(
                model, val_loader, scheduler, device, epoch,
                args.val_out_dir, T_LAT, H_LAT, W_LAT, args, accelerator,
                fast_mode=args.fast_validation
            )
        
        if epoch % args.save_interval == 0 and accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'sce_state_dict': unwrapped_model.sce.state_dict(),
                'injections_state_dict': [inj.state_dict() for inj in unwrapped_model.injections],
                'mpd_state_dict': unwrapped_model.mpd.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

    if args.wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()