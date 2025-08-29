import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset,DataLoader
from .config import FRAME_SIZE,CLIP_LENGTH,BATCH_SIZE,CLASSES

class SuspiciousActivityDataset(Dataset):
    def __init__(self, root_dir, clip_length=CLIP_LENGTH):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.sample = self._load_samples()
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self):
        samples = []
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            video_groups