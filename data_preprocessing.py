import glob
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    IMAGE_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
)

# ==========================================
# 1. Image Transforms / Preprocessing
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# 2. Data Exploration & Augmentation Preview
# ==========================================
def inspect_random_image(image_root_path=IMAGE_PATH):
    """Picks a random .jpg image from the dataset and prints its metadata."""
    random.seed(SEED)
    image_path_list = glob.glob(f"{image_root_path}/*/*/*/*.jpg")
    if not image_path_list:
        print(f"No images found matching '{image_root_path}/*/*/*/*.jpg'")
        return None

    random_image_path = random.choice(image_path_list)
    image_class = Path(random_image_path).parent.stem
    img = Image.open(random_image_path)

    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    return random_image_path


def plot_random_transforms(image_path, transform=train_transform, n=5):
    """Displays the original image alongside n augmented versions."""
    image = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, n + 1, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    mean = torch.tensor(IMAGENET_MEAN)
    std = torch.tensor(IMAGENET_STD)

    for i in range(n):
        transformed = transform(image)

        # De-normalize for visualization
        transformed = transformed.permute(1, 2, 0)
        transformed = transformed * std + mean
        transformed = transformed.clamp(0, 1)

        axes[i + 1].imshow(transformed)
        axes[i + 1].set_title(f"Augmented {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. Dataset & DataLoader Creation
# ==========================================
def get_datasets(train_dir=TRAIN_DIR, test_dir=TEST_DIR):
    """Creates and returns the training and testing ImageFolder datasets."""
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    print("Classes:", train_dataset.classes)
    print("Class to index:", train_dataset.class_to_idx)

    if len(train_dataset) > 0:
        image, label = train_dataset[0][0], train_dataset[0][1]
        print(f"Sample image shape: {image.shape}, dtype: {image.dtype}, label: {label} ({type(label)})")

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Creates and returns the training and testing DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    sample_img = inspect_random_image()
    if sample_img:
        plot_random_transforms(sample_img, train_transform, n=1)
    get_datasets()

