import os
import cv2
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class GenPropDataset(Dataset):
    def __init__(self, root_dir, resolution=576, clip_len=8, split='train'):
        """
        Args:
            root_dir (str): Path to the dataset root. Must contain 'original_videos', 
                            'edited_videos', and 'masks' subfolders.
            resolution (int): Height and width to resize frames to.
            clip_len (int): Number of frames to sample per video.
            split (str): 'train' or 'val'.
        """
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.clip_len = clip_len
        self.split = split

        self.orig_dir = self.root_dir / 'original_videos'
        self.edit_dir = self.root_dir / 'edited_videos'
        self.mask_dir = self.root_dir / 'masks'

        # Get list of common filenames (assuming filenames match across folders)
        self.video_files = sorted([f.name for f in self.orig_dir.glob('*.mp4')])
#        self.video_files = self.video_files[:600]
        print(f"Found {len(self.video_files)} videos")
        
        # Simple transform for normalization
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])  # Map [0,1] to [-1,1] for Latent Diffusion
        ])
        
        self.mask_transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.video_files)

    def _load_video_clip(self, path, start_idx, num_frames, is_mask=False):
        """Helper to load a sequence of frames from an MP4 using OpenCV."""
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        
        if start_idx >= total_frames:
            start_idx = 0
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8))
                continue
            
            if is_mask:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            frame = cv2.resize(frame, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            
            if is_mask:
                frame = Image.fromarray(frame, mode='L')
            else:
                frame = Image.fromarray(frame, mode='RGB')
                
            frames.append(frame)
            
        cap.release()
        return frames

    def __getitem__(self, idx):
        vid_name = self.video_files[idx]
        
        orig_path = self.orig_dir / vid_name
        edit_path = self.edit_dir / vid_name
        mask_path = self.mask_dir / vid_name

        # Determine total frames to pick a random start
        cap = cv2.VideoCapture(str(orig_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames > self.clip_len:
            start_idx = random.randint(0, total_frames - self.clip_len)
        else:
            start_idx = 0

        # Load clips
        orig_frames = self._load_video_clip(orig_path, start_idx, self.clip_len, is_mask=False)
        edit_frames = self._load_video_clip(edit_path, start_idx, self.clip_len, is_mask=False)
        mask_frames = self._load_video_clip(mask_path, start_idx, self.clip_len, is_mask=True)

        # Apply Transforms
        orig_tensors = torch.stack([self.transform(f) for f in orig_frames])  # [T, C, H, W]
        edit_tensors = torch.stack([self.transform(f) for f in edit_frames])  # [T, C, H, W]
        
        # Masks: Ensure binary (0 or 1)
        mask_tensors = torch.stack([self.mask_transform(f) for f in mask_frames])  # [T, 1, H, W]
        mask_tensors = (mask_tensors > 0.5).float()

        # v1_edited: First frame of edited video (for I2V conditioning)
        v1_edited = edit_tensors[0]  # [C, H, W]

        # Return: original video, edited video (full sequence), edited first frame, masks
        return orig_tensors, edit_tensors, v1_edited, mask_tensors