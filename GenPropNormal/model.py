import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableVideoDiffusionPipeline


class RegionAwareLoss(nn.Module):
    def __init__(self, lambda_mask=2.0, beta_grad=1.0, gamma_mpd=1.0):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.beta_grad = beta_grad
        self.gamma_mpd = gamma_mpd
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred, target, mask_latent, grad_feats=None,
                mask_pred=None, mask_target=None):
        pred32 = pred.float()
        target32 = target.float()
        mask_latent32 = mask_latent.float()

        inside_pred = mask_latent32 * pred32
        inside_target = mask_latent32 * target32
        L_mask = self.mse(inside_pred, inside_target)

        outside_mask = 1.0 - mask_latent32
        outside_pred = outside_mask * pred32
        outside_target = outside_mask * target32
        L_nonmask = self.mse(outside_pred, outside_target)

        if grad_feats is not None:
            grad_feats32 = grad_feats.float()
            mask_latent32_resized = mask_latent32
            
            if grad_feats32.shape[2:] != mask_latent32.shape[2:]:
                mask_latent32_resized = F.interpolate(
                    mask_latent32,
                    size=grad_feats32.shape[2:],
                    mode='nearest'
                )
            
            num_channels = grad_feats32.shape[1]
            grad_norm = torch.norm(grad_feats32, dim=1) / (num_channels ** 0.5)
            
            mask_squeezed = mask_latent32_resized.squeeze(1)
            L_grad = torch.mean(mask_squeezed * grad_norm)
        else:
            L_grad = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        if (mask_pred is not None) and (mask_target is not None):
            mpred32 = mask_pred.float()
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
            'L_mask': L_mask.item(),
            'L_grad': L_grad.item(),
            'L_mpd': L_mpd.item(),
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
        dtype = self.conv_in.weight.dtype
        device = latents_orig.device
        
        if latents_orig.dtype != dtype:
            latents_orig = latents_orig.to(dtype)

        B, C, T, H, W = latents_orig.shape
        
        t_emb = self.time_proj(timesteps.to(device))
        t_emb = t_emb.to(dtype)
        emb = self.time_embedding(t_emb)
        
        aug_emb = self.add_time_proj(added_time_ids.to(device).flatten())
        aug_emb = aug_emb.to(dtype)
        aug_emb = aug_emb.reshape(B, -1)
        aug_emb = self.add_embedding(aug_emb)

        temb = emb + aug_emb
        temb = temb.repeat_interleave(T, dim=0)

        x = latents_orig.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.conv_in(x)

        if image_only_indicator.ndim == 1:
            image_only_indicator = image_only_indicator.unsqueeze(-1).repeat(1, T)

        encoder_hidden_states = encoder_hidden_states.to(dtype)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(T, dim=0)

        feats = []
        for block in self.blocks:
            out = block(
                x, temb=temb,
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
    def __init__(self, in_channels, latent_time, latent_h, latent_w, hidden_channels=256):
        super().__init__()
        self.latent_time = latent_time
        self.latent_h = latent_h
        self.latent_w = latent_w
        
        # MLP: multiple layers with nonlinearities
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.GroupNorm(16, hidden_channels // 2),
            nn.GELU(),
            nn.Conv3d(hidden_channels // 2, 1, kernel_size=1),
        )
        
        # Standard initialization (Kaiming for layers before ReLU/GELU)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, feat):
        # feat: [B, C, T, H, W]
        x = feat.to(self.mlp[0].weight.dtype)
        x = torch.clamp(x, -10.0, 10.0)
        
        mask_logits = self.mlp(x)
        
        mask_logits = F.interpolate(
            mask_logits,
            size=(self.latent_time, self.latent_h, self.latent_w),
            mode="trilinear",
            align_corners=False,
        )
        return mask_logits


class InjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, out_channels)
        # Zero initialize everything so injection starts with no effect
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        x = x.to(self.conv.weight.dtype)
        out = self.conv(x)
        out = self.norm(out)
        return out


class GenPropModel(nn.Module):
    def __init__(self, backbone, sce, injections, mpd, image_encoder, feature_extractor,
                 grad_delta=0.01):
        super().__init__()
        self.vae = backbone.vae
        self.unet = backbone.unet
        self.backbone = backbone
        self.sce = sce
        self.injections = nn.ModuleList(injections)
        self.mpd = mpd
        self.image_encoder = image_encoder.float()
        self.feature_extractor = feature_extractor
        self.grad_delta = grad_delta
        
        # Hook-related attributes
        self._current_injected_feats = None
        self._injection_hooks = []
        self._mpd_features = None
        self._mpd_hook = None
    
    def _clear_hooks(self):
        """Remove all existing hooks"""
        for hook in self._injection_hooks:
            hook.remove()
        self._injection_hooks = []
        if self._mpd_hook is not None:
            self._mpd_hook.remove()
            self._mpd_hook = None
    
    def _register_injection_hooks(self):
        """Register injection hooks on down_blocks"""
        for idx, block in enumerate(self.unet.down_blocks[:len(self.injections)]):
            hook = block.register_forward_hook(self._make_injection_hook(idx))
            self._injection_hooks.append(hook)
    
    def _register_mpd_hook(self):
        """Register MPD hook on penultimate up_block"""
        num_up_blocks = len(self.unet.up_blocks)
        if num_up_blocks >= 2:
            penultimate_idx = num_up_blocks - 2
            self._mpd_hook = self.unet.up_blocks[penultimate_idx].register_forward_hook(
                self._mpd_feature_hook
            )
            return penultimate_idx
        return None
    
    def register_hooks(self):
        """Register all hooks - call after model is on device / after accelerator.prepare()"""
        self._clear_hooks()
        self._register_injection_hooks()
        penultimate_idx = self._register_mpd_hook()
        print(f"Registered {len(self._injection_hooks)} injection hooks, MPD hook on up_block[{penultimate_idx}]")
    
    def _mpd_feature_hook(self, module, inputs, output):
        """Capture output of penultimate up_block for MPD"""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self._mpd_features = hidden

    def _make_injection_hook(self, idx):
        def hook(module, inputs, output):
            if self._current_injected_feats is None:
                return output
            feat = self._current_injected_feats[idx]
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

    def get_clip_image_embeddings(self, images_pil):
        device = next(self.image_encoder.parameters()).device
        dtype = torch.float16
        
        pixel_values = self.feature_extractor(
            images=images_pil,
            return_tensors="pt"
        ).pixel_values.to(device, dtype=dtype)
        
        with torch.no_grad():
            img_embeds = self.image_encoder(pixel_values).image_embeds
        
        return img_embeds.unsqueeze(1)

    def encode_image_latents(self, images, noise_aug_strength=0.02, seed=None):
        device = images.device
        vae_dtype = self.vae.dtype
        B = images.shape[0]
        
        images_float = images.to(dtype=vae_dtype)
        
        if noise_aug_strength > 0:
            if seed is not None:
                g = torch.Generator(device=device).manual_seed(seed)
            else:
                g = None
            aug_noise = torch.randn(images_float.shape, device=device, dtype=vae_dtype, generator=g)
            images_aug = images_float + noise_aug_strength * aug_noise
        else:
            images_aug = images_float
        
        with torch.no_grad():
            latent_dist = self.vae.encode(images_aug).latent_dist
            cond_lat = latent_dist.mode()
        
        return cond_lat

    def _prepare_inputs_for_training(self, V_orig, v1_edited, t, T_LAT, H_LAT, W_LAT):
        """
        Prepare inputs for training.
        
        Args:
            V_orig: Original video (with object) [B, T, C, H, W]
            v1_edited: Edited first frame [B, C, H, W]
            t: Timesteps
            T_LAT, H_LAT, W_LAT: Latent dimensions
            
        Returns:
            latents_orig: Encoded original video [B, C, T, H, W]
            latents_v1_cond: Tiled edited first frame latents (for conditioning) [B, C, T, H, W]
            added_time_ids: Time conditioning
            encoder_hidden_states: CLIP embeddings
        """
        B, T, C, H, W = V_orig.shape
        dtype = self.unet.dtype
        vae_dtype = self.vae.dtype
        device = V_orig.device

        fps = 6
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        added_time_ids = torch.tensor(
            [fps, motion_bucket_id, noise_aug_strength],
            device=device, dtype=dtype
        )
        added_time_ids = added_time_ids.unsqueeze(0).repeat(B, 1)

        # CLIP embeddings from edited first frame
        images_resized = F.interpolate(v1_edited, size=(224, 224), mode='bicubic', align_corners=False)
        images_01 = (images_resized + 1.0) / 2.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        images_norm = (images_01.float() - mean) / std
        with torch.no_grad():
            encoder_hidden_states = self.image_encoder(images_norm).image_embeds.unsqueeze(1)

        # Encode original video (for SCE)
        V_flat = V_orig.view(B * T, C, H, W).to(dtype=vae_dtype)
        with torch.no_grad():
            lat_flat = self.vae.encode(V_flat).latent_dist.sample() * self.vae.config.scaling_factor
        _, C_lat, _, _ = lat_flat.shape
        latents_orig = lat_flat.view(B, T, C_lat, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)
        latents_orig = latents_orig.to(dtype=dtype)

        # Encode and tile edited first frame (for conditioning)
        v1_tiled = v1_edited.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, C, H, W).to(dtype=vae_dtype)
        with torch.no_grad():
            lat_v1_flat = self.vae.encode(v1_tiled).latent_dist.sample() * self.vae.config.scaling_factor
        latents_v1_cond = lat_v1_flat.view(B, T, C_lat, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)
        latents_v1_cond = latents_v1_cond.to(dtype=dtype)

        return latents_orig, latents_v1_cond, added_time_ids, encoder_hidden_states

    def _compute_sce_features(self, x_in, t, added_time_ids, image_only_indicator, encoder_hidden_states):
        """Helper to compute SCE features"""
        return self.sce(x_in, t, added_time_ids, image_only_indicator, encoder_hidden_states)

    def forward(self, V_orig, V_edited, v1_edited, t, noise, scheduler, mask_latent=None, injection_weight=1.0, 
                compute_grad_loss=True):
        """
        Args:
            V_orig: Original video (with object) [B, T, C, H, W] - fed to SCE
            V_edited: Edited video (without object) [B, T, C, H, W] - target for denoising
            v1_edited: Edited first frame [B, C, H, W] - conditioning for I2V
            t: Timesteps [B]
            noise: Random noise [B, 4, T, H_LAT, W_LAT]
            scheduler: Diffusion scheduler (needed for v-prediction target computation)
            mask_latent: Ground truth mask in latent space
            injection_weight: Weight for SCE injection
            compute_grad_loss: Whether to compute gradient loss
        """
        B, T, C, H, W = V_orig.shape
        _, _, _, H_LAT, W_LAT = noise.shape
        dtype = self.unet.dtype
        device = V_orig.device

        latents_orig, latents_v1_cond, added_time_ids, encoder_hidden_states = \
            self._prepare_inputs_for_training(V_orig, v1_edited, t, T, H_LAT, W_LAT)
        
        # Encode the full edited video sequence (this is the target)
        vae_dtype = self.vae.dtype
        V_edited_flat = V_edited.view(B * T, C, H, W).to(dtype=vae_dtype)
        with torch.no_grad():
            latents_edited_flat = self.vae.encode(V_edited_flat).latent_dist.sample() * self.vae.config.scaling_factor
        _, C_lat, _, _ = latents_edited_flat.shape
        latents_edited = latents_edited_flat.view(B, T, C_lat, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)
        latents_edited = latents_edited.to(dtype=dtype)
        
        # Compute velocity target for v-prediction
        # v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * sample
        alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
        
        # Reshape for broadcasting [B] -> [B, 1, 1, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1, 1)
        
        # Velocity target
        velocity_target = sqrt_alpha_prod * noise.to(dtype=dtype) - sqrt_one_minus_alpha_prod * latents_edited
        
        # Store for returning
        self._velocity_target = velocity_target

        # SCE input: concatenate original video latents with conditioning latents
        x_in = torch.cat([latents_orig, latents_v1_cond], dim=1)
        image_only_indicator = torch.zeros(B, dtype=torch.long, device=device)

        sce_feats = self._compute_sce_features(
            x_in, t, added_time_ids, image_only_indicator, encoder_hidden_states
        )

        # Compute gradient features using finite difference if requested
        grad_feats = None
        if compute_grad_loss and mask_latent is not None:
            mask_expanded = mask_latent.expand(-1, 4, -1, -1, -1)
            delta = self.grad_delta * torch.cat([mask_expanded, mask_expanded], dim=1)
            x_in_perturbed = x_in + delta.to(dtype=x_in.dtype)
            
            with torch.no_grad():
                sce_feats_perturbed = self._compute_sce_features(
                    x_in_perturbed, t, added_time_ids, image_only_indicator, encoder_hidden_states
                )
            
            grad_feats = (sce_feats_perturbed[-1] - sce_feats[-1]) / self.grad_delta
            
            if grad_feats.shape[2:] != mask_latent.shape[2:]:
                grad_feats = F.interpolate(
                    grad_feats,
                    size=mask_latent.shape[2:],
                    mode='nearest'
                )

        # Prepare injection features
        injected_feats = []
        for inj, feat in zip(self.injections, sce_feats):
            inj_feat = inj(feat)
            if injection_weight != 1.0:
                inj_feat = inj_feat * injection_weight
            injected_feats.append(inj_feat)

        self._current_injected_feats = injected_feats

        # KEY FIX: Noise the full edited video latents, not tiled first frame
        noisy_latents = latents_edited + noise.to(dtype=dtype)
        # Concatenate with conditioning (tiled edited first frame) for UNet input
        unet_input_latents = torch.cat([noisy_latents, latents_v1_cond], dim=1)
        noisy_latents_for_unet = unet_input_latents.permute(0, 2, 1, 3, 4)

        # Reset MPD features before UNet forward
        self._mpd_features = None
        
        unet_out = self.unet(
            noisy_latents_for_unet,
            t,
            encoder_hidden_states=encoder_hidden_states.to(dtype=dtype),
            added_time_ids=added_time_ids.to(dtype=dtype),
        )
        pred_noise = unet_out.sample if hasattr(unet_out, "sample") else unet_out
        pred_noise = pred_noise.view(B, T, 4, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)

        self._current_injected_feats = None
        
        # Get MPD prediction from UNet decoder features (captured by hook)
        if self._mpd_features is None:
            raise RuntimeError("MPD features not captured. The hook on penultimate up_block failed. "
                             "Make sure to call model.register_hooks() after accelerator.prepare()")
        
        # _mpd_features shape: [B*T, C, H, W] -> reshape to [B, C, T, H, W]
        BT, C_mpd, H_mpd, W_mpd = self._mpd_features.shape
        mpd_input = self._mpd_features.view(B, T, C_mpd, H_mpd, W_mpd).permute(0, 2, 1, 3, 4)
        mask_pred = self.mpd(mpd_input)
        self._mpd_features = None  # Clear after use

        return pred_noise, mask_pred, grad_feats, self._velocity_target

    def inference_step(self, latents, cond_latents, encoder_hidden_states, 
                       added_time_ids, t, do_cfg=True, guidance_scale=3.0,
                       V_orig=None, v1_edited=None, injection_weight=1.0):
        """Single denoising step for inference."""
        B, C, T, H_LAT, W_LAT = latents.shape
        device = latents.device
        dtype = self.unet.dtype
        
        latents = latents.to(dtype=dtype)
        cond_latents = cond_latents.to(dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype)
        added_time_ids = added_time_ids.to(dtype=dtype)
        
        if do_cfg:
            lat_in = torch.cat([latents, latents], dim=0)
            cond_lat_uncond = torch.zeros_like(cond_latents)
            cond_lat_in = torch.cat([cond_lat_uncond, cond_latents], dim=0)
            neg_embeds = torch.zeros_like(encoder_hidden_states)
            embeds_in = torch.cat([neg_embeds, encoder_hidden_states], dim=0)
            time_ids_in = added_time_ids.repeat(2, 1)
            if isinstance(t, (int, float)):
                t_in = torch.tensor([t, t], device=device)
            elif t.ndim == 0:
                t_in = t.unsqueeze(0).repeat(2)
            else:
                t_in = t.repeat(2) if t.shape[0] == 1 else torch.cat([t, t])
        else:
            lat_in = latents
            cond_lat_in = cond_latents
            embeds_in = encoder_hidden_states
            time_ids_in = added_time_ids
            if isinstance(t, (int, float)):
                t_in = torch.tensor([t], device=device)
            elif t.ndim == 0:
                t_in = t.unsqueeze(0)
            else:
                t_in = t
        
        if injection_weight > 0 and V_orig is not None and v1_edited is not None:
            B_orig = V_orig.shape[0]
            T_orig = V_orig.shape[1]
            vae_dtype = self.vae.dtype
            
            V_flat = V_orig.view(B_orig * T_orig, 3, V_orig.shape[3], V_orig.shape[4]).to(dtype=vae_dtype)
            with torch.no_grad():
                lat_orig_flat = self.vae.encode(V_flat).latent_dist.mode()
            latents_orig = lat_orig_flat.view(B_orig, T_orig, 4, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)
            latents_orig = latents_orig.to(dtype=dtype)
            
            cond_for_sce = cond_latents[:B_orig, :, :T_orig, :, :].to(dtype=dtype)
            
            x_in_sce = torch.cat([latents_orig, cond_for_sce], dim=1)
            image_only_indicator = torch.zeros(B_orig, dtype=torch.long, device=device)
            
            t_sce = t_in[:B_orig] if t_in.shape[0] >= B_orig else t_in
            sce_feats = self.sce(
                x_in_sce, t_sce,
                added_time_ids[:B_orig], image_only_indicator, 
                encoder_hidden_states[:B_orig]
            )
            
            injected_feats = []
            for inj, feat in zip(self.injections, sce_feats):
                inj_feat = inj(feat) * injection_weight
                if do_cfg:
                    inj_feat = torch.cat([torch.zeros_like(inj_feat), inj_feat], dim=0)
                injected_feats.append(inj_feat)
            self._current_injected_feats = injected_feats
        else:
            self._current_injected_feats = None
        
        unet_in = torch.cat([lat_in, cond_lat_in], dim=1)
        unet_in = unet_in.permute(0, 2, 1, 3, 4)
        
        noise_pred = self.unet(
            unet_in,
            t_in,
            encoder_hidden_states=embeds_in,
            added_time_ids=time_ids_in,
        ).sample
        
        self._current_injected_feats = None
        
        batch_size = 2 * B if do_cfg else B
        noise_pred = noise_pred.view(batch_size, T, 4, H_LAT, W_LAT).permute(0, 2, 1, 3, 4)
        
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guidance = torch.linspace(1.0, guidance_scale, T, device=device, dtype=dtype)
            guidance = guidance.view(1, 1, T, 1, 1)
            noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
        
        return noise_pred