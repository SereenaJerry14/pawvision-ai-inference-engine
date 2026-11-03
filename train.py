import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config import (
    DEVICE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    OUTPUT_DIR,
    WEIGHT_DECAY,
)
from data_preprocessing import get_dataloaders, get_datasets
from model import build_model


def train_model(
    model,
    train_loader,
    test_loader,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    save_path=MODEL_SAVE_PATH,
):
    """Trains the model across num_epochs, evaluates on test_loader, and saves weights."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100.0 * val_correct / val_total
        val_loss = val_loss / len(test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=OUTPUT_DIR):
    """Plots and saves training/testing loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # -------- LOSS PLOT --------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.grid(True)

    # -------- ACCURACY PLOT --------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "Accuracy&loss.jpg")
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.show()


def plot_confusion_matrix(model, test_loader, class_names, device=DEVICE, save_dir=OUTPUT_DIR):
    """Computes, plots, and saves the confusion matrix on the test dataset."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.grid(False)

    save_path = os.path.join(save_dir, "Confusion matrix.jpg")
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    print(f"Saved confusion matrix to {save_path}")
    plt.show()


if __name__ == "__main__":
    print("Using device:", DEVICE)
    train_ds, test_ds = get_datasets()
    train_dl, test_dl = get_dataloaders(train_ds, test_ds)

    cnn_model = build_model(DEVICE)
    t_losses, v_losses, t_accs, v_accs = train_model(cnn_model, train_dl, test_dl, DEVICE)

    plot_training_metrics(t_losses, v_losses, t_accs, v_accs)
    plot_confusion_matrix(cnn_model, test_dl, train_ds.classes, DEVICE)

