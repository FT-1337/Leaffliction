#!/usr/bin/env python3
"""
predict.py
==========

Part 4 – Use the trained plant disease classifier to predict the class
of a single leaf image.

- Loads the model saved by train.py (leaf_classifier.joblib).
- Reads the given image.
- Applies the same HSV histogram feature extraction as in training.
- Predicts the disease/health class.
- Prints the predicted class and confidence.
- Displays the original image and the transformed (resized) image side by side.
- Saves the visualization in a 'predictions/' directory.

Example usage

-------------
    ./predict.py ./Apple/apple_healthy/image\\ \\(1\\).JPG
    ./predict.py ./Apple/apple_healthy/image.JPG \
        --model leaffliction_model/leaf_classifier.joblib
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import load


def extract_features(img: np.ndarray,
                     size=(128, 128),
                     bins=32) -> np.ndarray:
    """
    Same feature extractor as in train.py:
    - Resize
    - Convert to HSV
    - Per-channel histograms
    """
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    features = []
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())

    return np.array(features, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict plant disease class for a single leaf image.",
    )
    parser.add_argument(
        "image_path",
        help="Path to the leaf image to classify.",
    )
    parser.add_argument(
        "--model",
        default="leaffliction_model/leaf_classifier.joblib",
        help="Path to the trained model (.joblib) produced by train.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions",
        help="Directory to save visualization images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model)
    out_dir = Path(args.output_dir)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run train.py first or specify --model PATH_TO_MODEL.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model bundle
    bundle = load(model_path)
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    image_size = tuple(bundle.get("image_size", (128, 128)))
    bins = int(bundle.get("bins", 32))

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        sys.exit(1)

    # Extract features
    feats = extract_features(img, size=image_size, bins=bins)
    feats = feats.reshape(1, -1)

    # Predict
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]

    pred_idx = model.predict(feats)[0]
    # If labels were encoded as integers, invert them
    try:
        label = encoder.inverse_transform([pred_idx])[0]
    except Exception:
        label = str(pred_idx)

    print("\n" + "=" * 60)
    print("🌿 PLANT DISEASE PREDICTOR")
    print("=" * 60)
    print(f"🖼️  Image: {image_path}")
    print(f"🔮 Predicted class: {label}")

    if probs is not None:
        # Confidence of predicted class
        class_index = list(encoder.classes_).index(label)
        confidence = probs[class_index]
        print(f"📊 Confidence: {confidence * 100:.2f}%")

    print("=" * 60 + "\n")

    # Create side-by-side visualization (original vs transformed)
    img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img_show_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_show_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_show_orig)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_show_resized)
    axes[1].set_title(
        f"Transformed\nPred: {label}",
        fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()

    base = image_path.stem.replace(" ", "_")
    out_img_path = out_dir / f"prediction_{base}.png"
    plt.savefig(out_img_path, dpi=200)
    plt.show()   # display window during evaluation
    plt.close()

    print("Visualization saved to:", out_img_path)


if __name__ == "__main__":
    main()
