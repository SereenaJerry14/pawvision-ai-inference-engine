import os
import torch

# ==========================================
# Hyperparameters & Paths Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Dataset & Output Paths
IMAGE_PATH = "archive"
TRAIN_DIR = os.path.join(IMAGE_PATH, "training_set", "training_set")
TEST_DIR = os.path.join(IMAGE_PATH, "test_set", "test_set")

OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "cat_dog_model.pth")
PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")

# ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

