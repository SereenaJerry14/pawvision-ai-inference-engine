import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from config import (
    BATCH_SIZE,
    DEVICE,
    MODEL_SAVE_PATH,
    PREDICTIONS_CSV_PATH,
    TEST_DIR,
)
from data_preprocessing import inspect_random_image, test_transform
from model import load_trained_model


def predict_single_image(model, image_path, device=DEVICE, transform=test_transform):
    """Runs inference on a single image and displays the prediction."""
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ["Cat", "Dog"]
    predicted_class = classes[predicted.item()]
    confidence_pct = confidence.item() * 100

    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class} ({confidence_pct:.2f}%)")
    plt.axis("off")
    plt.show()

    print("Prediction:", predicted_class)
    print(f"Confidence: {confidence_pct:.2f}%")
    return predicted_class, confidence_pct


def evaluate_and_export_predictions(
    model,
    test_dataset,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    csv_path=PREDICTIONS_CSV_PATH,
):
    """Evaluates the model on the test set, saves predictions to CSV, and plots top 5 wrong predictions."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = test_dataset.classes

    results = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)

            for i in range(len(labels)):
                results.append({
                    "Image_Path": test_dataset.samples[len(results)][0],
                    "Real_Label": class_names[labels[i].item()],
                    "Predicted_Label": class_names[preds[i].item()],
                    "Confidence (%)": confidence[i].item() * 100,
                })

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved as {csv_path}")

    print("Total test images:", len(test_dataset))
    print(df.head())

    wrong_df = df[df["Real_Label"] != df["Predicted_Label"]]
    print("Total Wrong Predictions:", len(wrong_df))
    print(wrong_df.head())

    accuracy = (df["Real_Label"] == df["Predicted_Label"]).mean() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot top 5 highest-confidence wrong predictions
    if not wrong_df.empty:
        top_wrong = wrong_df.sort_values(by="Confidence (%)", ascending=False).head(5)

        plt.figure(figsize=(15, 5))
        for i in range(len(top_wrong)):
            row = top_wrong.iloc[i]
            img = Image.open(row["Image_Path"]).convert("RGB")

            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(
                f"Real: {row['Real_Label']}\n"
                f"Pred: {row['Predicted_Label']}\n"
                f"Conf: {row['Confidence (%)']:.2f}%"
            )
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    return df, wrong_df


if __name__ == "__main__":
    print("Loading trained model from:", MODEL_SAVE_PATH)
    trained_model = load_trained_model(MODEL_SAVE_PATH, DEVICE)

    sample_image = inspect_random_image()
    if sample_image:
        predict_single_image(trained_model, sample_image, DEVICE)

    test_ds = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    evaluate_and_export_predictions(trained_model, test_ds, DEVICE)

