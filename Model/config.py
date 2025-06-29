import os
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset configuration
UCF_ROOT = "UCF_Crimes"
CLASSES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

# Model parameters
FRAME_SIZE = (64, 64)  # Input image size
CLIP_LENGTH = 16        # Number of frames per clip
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# checkpoint configuration for faster debugging
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)