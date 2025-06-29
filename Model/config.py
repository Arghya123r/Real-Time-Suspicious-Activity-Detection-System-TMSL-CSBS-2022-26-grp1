import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UCF_ROOT = "UCF_CRIMES"