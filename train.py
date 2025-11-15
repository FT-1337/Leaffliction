#!/usr/bin/env python3
"""
train.py
========

Part 4 – Plant disease classifier training script for the Leaffliction project.

What it does
------------
- Takes a root directory of images (each subdirectory is a class).
- Fetches all images in the subdirectories.
- Applies light data augmentation (flip, rotations, brightness).
- Saves the augmented images into an output directory.
- Extracts simple HSV color histogram features from each (augmented) image.
- Splits the dataset into Training and Validation sets.
- Trains a RandomForest classifier on the Training set.
- Evaluates it on the Validation set and prints accuracy + classification report.
- Saves the model + metadata.
- Creates a .zip archive that includes:
    * The trained model
    * Metadata
    * All increased/modified (augmented) images

Example usage
-------------
    ./train.py ./Apple
    ./train.py ./images --output-dir ./leaffliction_model
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".gif", ".bmp")


def log(msg: str) -> None:
    """Print a message and flush (so logs appear immediately)."""
    print(msg, flush=True)


def find_images_by_class(root_dir: Path):
    """
    Return a dict: {class_name: [list of image Paths]}.

    Each subdirectory under root_dir is treated as a class.
    """
    data = defaultdict(list)

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        class_name = subdir.name
        for ext in SUPPORTED_FORMATS:
            for p in subdir.glob(f"*{ext}"):
                data[class_name].append(p)
            for p in subdir.glob(f"*{ext.upper()}"):
                data[class_name].append(p)

    # Remove empty classes
    return {k: v for k, v in data.items() if v}


def extract_features(img: np.ndarray,
                     size=(128, 128),
                     bins=32) -> np.ndarray:
    """
    Extract simple color histogram features from an image.

    Steps:
      - Resize to `size`
      - Convert to HSV
      - Compute 1D histograms for each channel (H, S, V)
      - Normalize and flatten
    """
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    features = []

    for i in range(3):  # H, S, V
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())

    return np.array(features, dtype=np.float32)


def augment_image(img: np.ndarray):
    """
    Returns a dict of augmented versions of the image.

    Keys are suffixes used for filenames.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Horizontal flip
    flip = cv2.flip(img, 1)

    # Rotations around the center
    M1 = cv2.getRotationMatrix2D(center, 20, 1.0)
    rot_p = cv2.warpAffine(img, M1, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    M2 = cv2.getRotationMatrix2D(center, -20, 1.0)
    rot_n = cv2.warpAffine(img, M2, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Brightness jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_br = hsv.copy()
    factor = 1.2
    hsv_br[:, :, 2] = np.clip(hsv_br[:, :, 2] * factor, 0, 255)
    bright = cv2.cvtColor(hsv_br, cv2.COLOR_HSV2BGR)

    return {
        "orig": img,
        "flip": flip,
        "rot_p": rot_p,
        "rot_n": rot_n,
        "bright": bright,
    }


def build_dataset(images_by_class,
                  augmented_dir: Path,
                  feature_size=(128, 128),
                  bins=32):
    """
    For each image:
      - Load it
      - Create augmented variants
      - Save augmented images to augmented_dir
      - Extract features + labels

    Returns:
      X: np.ndarray of features
      y: np.ndarray of labels (string class names)
    """
    X = []
    y = []

    augmented_dir.mkdir(parents=True, exist_ok=True)

    for class_name, paths in sorted(images_by_class.items()):
        class_out_dir = augmented_dir / class_name
        class_out_dir.mkdir(exist_ok=True)
        log(f"  📁 {class_name} ({len(paths)} original images)")

        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is None:
                log(f"     ⚠️ Could not read image: {img_path}")
                continue

            base = img_path.stem

            variants = augment_image(img)
            for suffix, aug_img in variants.items():
                filename = f"{base}_{suffix}.png"
                save_path = class_out_dir / filename

                cv2.imwrite(str(save_path), aug_img)

                feats = extract_features(
                    aug_img,
                    size=feature_size,
                    bins=bins,
                )
                X.append(feats)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    return X, y


def train_classifier(X, y, random_state=42):
    """Train a RandomForest classifier and return it."""
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def create_zip(output_dir: Path, zip_name: str):
    """
    Create a zip archive of the output directory.

    Example:
      output_dir="leaffliction_model", zip_name="leaffliction_learnings"
      → leaffliction_learnings.zip (next to output_dir)
    """
    root = output_dir.resolve()
    base_name = Path(zip_name).with_suffix("").name
    archive_path = root.parent / base_name
    shutil.make_archive(str(archive_path), "zip", root_dir=root)
    return archive_path.with_suffix(".zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a plant disease classifier from an image dataset.",
    )
    parser.add_argument(
        "data_directory",
        help="Root directory containing subdirectories for each class.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="leaffliction_model",
        help="Directory to store model, metadata and augmented dataset.",
    )
    parser.add_argument(
        "--zip-name",
        default="leaffliction_learnings",
        help="Base name for the zip archive of learnings (without .zip).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (default: 0.2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_directory)
    if not data_dir.exists() or not data_dir.is_dir():
        log(f"\n❌ Provided path is not a directory: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    augmented_dir = output_dir / "augmented_dataset"
    model_dir = output_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 70)
    log("🌿 PLANT DISEASE CLASSIFIER – TRAINING")
    log("=" * 70)

    # 1. Scan images
    log(f"\n🔍 Scanning dataset: {data_dir}")
    images_by_class = find_images_by_class(data_dir)
    if not images_by_class:
        log("❌ No images found in subdirectories.")
        sys.exit(1)

    total_original = sum(len(v) for v in images_by_class.values())
    log(f"✅ Found {len(images_by_class)} classes, {total_original} original images.\n")

    # 2. Build dataset with augmentations
    log("🧪 Generating augmented dataset and extracting features...")
    X, y_labels = build_dataset(
        images_by_class,
        augmented_dir=augmented_dir,
        feature_size=(128, 128),
        bins=32,
    )

    log(f"\n✅ Feature matrix shape: {X.shape}")
    log(f"✅ Total augmented samples: {len(y_labels)}")

    # 3. Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    class_names = list(encoder.classes_)
    log("\n📚 Classes:")
    for idx, name in enumerate(class_names):
        log(f"   {idx}: {name}")

    # 4. Train/validation split
    log("\n🔀 Splitting into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    log(f"   Training samples:   {X_train.shape[0]}")
    log(f"   Validation samples: {X_val.shape[0]}")

    if X_val.shape[0] < 100:
        log("⚠️ Validation set has fewer than 100 images. "
            "For the evaluation, make sure you run with enough data.")

    # 5. Train classifier
    log("\n🚀 Training RandomForest classifier...")
    clf = train_classifier(X_train, y_train)
    log("✅ Training complete.")

    # 6. Evaluate
    log("\n📈 Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    log(f"   Accuracy: {acc * 100:.2f}%")

    report = classification_report(
        y_val,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    log("\n🔎 Classification report:")
    log(report)

    # 7. Save model and metadata
    model_path = model_dir / "leaf_classifier.joblib"
    metadata_path = model_dir / "metadata.json"
    report_path = model_dir / "validation_report.txt"

    log("\n💾 Saving model, metadata and report...")
    model_bundle = {
        "model": clf,
        "label_encoder": encoder,
        "image_size": (128, 128),
        "bins": 32,
    }
    dump(model_bundle, model_path)

    metadata = {
        "classes": class_names,
        "accuracy": float(acc),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "original_images": int(total_original),
        "augmented_dir": str(augmented_dir),
        "test_size": args.test_size,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Validation classification report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    log(f"✅ Model saved to:       {model_path}")
    log(f"✅ Metadata saved to:    {metadata_path}")
    log(f"✅ Report saved to:      {report_path}")

    # 8. Create zip archive of learnings + augmented images
    log("\n📦 Creating zip archive with learnings + augmented dataset...")
    zip_path = create_zip(output_dir, args.zip_name)
    log(f"✅ Archive created: {zip_path}")

    log("\n" + "=" * 70)
    log("✨ TRAINING FINISHED")
    log("=" * 70 + "\n")


if __name__ == "__main__":
    main()

