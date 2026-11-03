import os

from config import DEVICE, IMAGE_PATH, MODEL_SAVE_PATH, NUM_EPOCHS, OUTPUT_DIR
from data_preprocessing import (
    get_dataloaders,
    get_datasets,
    inspect_random_image,
    plot_random_transforms,
    train_transform,
)
from model import build_model
from predict import evaluate_and_export_predictions, predict_single_image
from train import plot_confusion_matrix, plot_training_metrics, train_model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Using device:", DEVICE)

    # 1. Data Exploration & Augmentation Visualization
    sample_image = inspect_random_image(IMAGE_PATH)
    if sample_image:
        plot_random_transforms(sample_image, train_transform, n=1)

    # 2. Data Loading & Preprocessing
    train_dataset, test_dataset = get_datasets()
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

    # 3. Model Initialization
    model = build_model(DEVICE)

    # 4. Training & Evaluation Plots
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH,
    )
    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(model, test_loader, train_dataset.classes, DEVICE)

    # 5. Prediction & Error Analysis
    if sample_image:
        predict_single_image(model, sample_image, DEVICE)

    evaluate_and_export_predictions(model, test_dataset, DEVICE)


if __name__ == "__main__":
    main()

