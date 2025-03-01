from tensorflow.keras.models import load_model  # type:ignore
from sklearn.metrics import classification_report, confusion_matrix  # type:ignore
import numpy as np
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_data import read_params  # Corrected function call
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type:ignore


def evaluate(config_path):
    config = read_params(config_path)  # Correct function to read YAML config

    # Load image augmentation parameters
    batch_size = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    rescale = config['img_augment']['rescale']
    shear_range = config['img_augment']['shear_range']
    zoom_range = config['img_augment']['zoom_range']
    horizontal_flip = config['img_augment']['horizontal_flip']
    vertical_flip = config['img_augment']['vertical_flip']

    # Load paths from config
    test_path = config['model']['test_path']
    model_dir = config['model']['sav_dir']  # This should be a directory, not a file
    model_path = os.path.join(model_dir, "trained.h5")  # Ensure correct file path

    # Convert image size to tuple
    image_size = tuple(config["model"]["image_size"])

    # Check if the model file exists before loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load trained model
    model = load_model(model_path)
    print("Model Loaded Successfully!")

    # Create ImageDataGenerator for testing
    test_gen = ImageDataGenerator(
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip
    )

    # Create test dataset
    test_set = test_gen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False  # Ensure order is maintained for evaluation
    )

    # Get class indices mapping
    label_map = test_set.class_indices
    target_names = list(label_map.keys())  # Dynamically get class names

    # Predict on test set
    print("Running predictions on test set...")
    y_pred = model.predict(test_set)
    y_pred = np.argmax(y_pred, axis=1)

    # Generate reports directory if it doesn't exist
    reports_dir = "reports/figures/"
    os.makedirs(reports_dir, exist_ok=True)

    # Generate and save confusion matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(test_set.classes, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'))
    plt.close()
    print("Confusion Matrix saved successfully!")

    # Generate classification report
    print("Generating Classification Report...")
    report = classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True)

    # Save classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(reports_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path, index=True)
    print(f"Classification Report saved at {report_csv_path}")

    print("Evaluation completed. Results saved in 'reports/figures/'.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="params.yaml")
    parsed_args = args.parse_args()

    evaluate(config_path=parsed_args.config)
