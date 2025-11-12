#!/usr/bin/env python3
"""
DataAugmentation.py
===================

A comprehensive data augmentation tool for balancing plant disease datasets.

Purpose:
- Applies 6 types of augmentations to plant images to balance unequal datasets
- Creates variations of each image to increase training data diversity
- Generates augmented images with standardized naming convention

Input:
    - Directory path containing subdirectories of plant images
    - Each subdirectory represents a plant type/disease category
    - Supported formats: .jpg, .jpeg, .png, .gif, .bmp

Output:
    - 6 augmented versions of each image saved in the same directory
    - Naming convention: original_filename_(augmentation_type).JPG
    - Examples: image (1)_Flip.JPG, image (1)_Rotate.JPG, etc.

Augmentation Types:
    1. Flip - Mirror the image horizontally
    2. Rotate - Rotate the image at a random angle (15-45 degrees)
    3. Skew - Apply perspective skew transformation
    4. Shear - Slant the image using shear transformation
    5. Crop - Crop a random portion of the image and resize
    6. Distortion - Apply brightness and contrast adjustments

Example Usage:
    python DataAugmentation.py ./images
    python DataAugmentation.py /path/to/plant/images/directory

Project Structure:
    ├── images/                  # Input/Output: Plant image directories
    │   ├── Apple_Black_rot/
    │   ├── Apple_healthy/
    │   ├── Grape_Black_rot/
    │   └── ...
    ├── DataAugmentation.py      # This program
    ├── Distribution.py          # Related program for analysis
    ├── requirements.txt         # Dependencies
    └── venv/                    # Virtual environment
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random


class DataAugmentor:
    """
    Augments plant images using 6 different transformation techniques.
    
    Attributes:
        data_directory (Path): Path to the images directory
        supported_formats (tuple): Tuple of supported image file extensions
        augmentation_types (list): List of augmentation method names
    """
    
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    AUGMENTATION_TYPES = ['Flip', 'Rotate', 'Skew', 'Shear', 'Crop', 'Distortion']
    
    def __init__(self, directory_path):
        """
        Initialize the augmentor with a directory path.
        
        Args:
            directory_path (str): Path to the directory containing plant images
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
            NotADirectoryError: If the path is not a directory
        """
        self.data_directory = Path(directory_path)
        
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not self.data_directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")
    
    def get_all_images(self):
        """
        Scan the directory recursively and find all images in subdirectories.
        
        Returns:
            dict: Dictionary with subdirectory names as keys and lists of image paths as values
        """
        images_by_category = {}
        subdirectories = [d for d in self.data_directory.iterdir() if d.is_dir()]
        
        for subdir in sorted(subdirectories):
            image_paths = []
            for format_ext in self.SUPPORTED_FORMATS:
                image_paths.extend(subdir.glob(f'*{format_ext}'))
                image_paths.extend(subdir.glob(f'*{format_ext.upper()}'))
            
            if image_paths:
                images_by_category[subdir.name] = sorted(image_paths)
        
        return images_by_category
    
    def flip_image(self, image):
        """
        Flip the image horizontally (mirror).
        
        Args:
            image (PIL.Image): Image to flip
            
        Returns:
            PIL.Image: Flipped image
        """
        return ImageOps.mirror(image)
    
    def rotate_image(self, image):
        """
        Rotate the image at a random angle (15-45 degrees).
        
        Args:
            image (PIL.Image): Image to rotate
            
        Returns:
            PIL.Image: Rotated image
        """
        angle = random.uniform(15, 45)
        return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    def skew_image(self, image):
        """
        Apply perspective skew transformation to the image.
        
        Args:
            image (PIL.Image): Image to skew
            
        Returns:
            PIL.Image: Skewed image
        """
        width, height = image.size
        # Define coefficients for perspective transformation
        # This creates a slight trapezoidal distortion
        coefficient = 0.2
        xshift = int(width * coefficient)
        
        points = [
            (0, 0), (width, 0),
            (0, height), (width, height)
        ]
        
        new_points = [
            (xshift, 0), (width - xshift, 0),
            (0, height), (width, height)
        ]
        
        # Create transformation coefficients
        coeffs = find_coeffs(new_points, points)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    def shear_image(self, image):
        """
        Apply shear transformation to the image.
        
        Args:
            image (PIL.Image): Image to shear
            
        Returns:
            PIL.Image: Sheared image
        """
        width, height = image.size
        shear_factor = 0.3
        
        points = [
            (0, 0), (width, 0),
            (0, height), (width, height)
        ]
        
        new_points = [
            (0, 0), (width, int(height * shear_factor)),
            (0, height), (width, int(height * (1 - shear_factor)))
        ]
        
        coeffs = find_coeffs(new_points, points)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    def crop_image(self, image):
        """
        Crop a random portion of the image and resize back to original size.
        
        Args:
            image (PIL.Image): Image to crop
            
        Returns:
            PIL.Image: Cropped and resized image
        """
        width, height = image.size
        crop_factor = 0.8  # Crop to 80% of original size
        
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        right = left + new_width
        bottom = top + new_height
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)
    
    def distortion_image(self, image):
        """
        Apply brightness and contrast adjustments (color distortion).
        
        Args:
            image (PIL.Image): Image to distort
            
        Returns:
            PIL.Image: Distorted image
        """
        # Adjust brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        brightness_factor = random.uniform(0.7, 1.3)
        image = brightness_enhancer.enhance(brightness_factor)
        
        # Adjust contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        contrast_factor = random.uniform(0.8, 1.2)
        image = contrast_enhancer.enhance(contrast_factor)
        
        return image
    
    def augment_image(self, image_path):
        """
        Generate 6 augmented versions of an image and save them.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            bool: True if augmentation successful, False otherwise
        """
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get the base filename without extension
            base_name = image_path.stem
            parent_dir = image_path.parent
            
            # Create augmented versions
            augmentations = {
                'Flip': self.flip_image(image),
                'Rotate': self.rotate_image(image),
                'Skew': self.skew_image(image),
                'Shear': self.shear_image(image),
                'Crop': self.crop_image(image),
                'Distortion': self.distortion_image(image)
            }
            
            # Save augmented images
            for aug_type, aug_image in augmentations.items():
                output_filename = f"{base_name}_{aug_type}.JPG"
                output_path = parent_dir / output_filename
                aug_image.save(output_path, 'JPEG', quality=95)
            
            return True
        
        except Exception as e:
            print(f"❌ Error augmenting {image_path}: {e}")
            return False
    
    def run_augmentation(self):
        """
        Execute the complete augmentation pipeline for all images.
        """
        print("\n" + "=" * 70)
        print("🖼️  PLANT IMAGE DATA AUGMENTOR")
        print("=" * 70)
        
        images_by_category = self.get_all_images()
        
        if not images_by_category:
            print("⚠️  No images found in subdirectories.")
            return False
        
        total_original = 0
        total_augmented = 0
        
        for category, image_paths in sorted(images_by_category.items()):
            print(f"\n📁 Processing: {category}")
            print("-" * 70)
            
            category_count = 0
            for image_path in image_paths:
                if self.augment_image(image_path):
                    category_count += 1
                    print(f"  ✅ {image_path.name}")
            
            augmented_count = category_count * len(self.AUGMENTATION_TYPES)
            print(f"  📊 {category_count} images augmented → {augmented_count} new images created")
            
            total_original += category_count
            total_augmented += augmented_count
        
        print("\n" + "=" * 70)
        print("📈 AUGMENTATION SUMMARY")
        print("=" * 70)
        print(f"Original images processed: {total_original}")
        print(f"Augmentation types per image: {len(self.AUGMENTATION_TYPES)}")
        print(f"Total augmented images created: {total_augmented}")
        print(f"Total images after augmentation: {total_original + total_augmented}")
        print("=" * 70)
        print("✨ Data augmentation completed successfully!\n")
        
        return True


def find_coeffs(source_coords, target_coords):
    """
    Find the perspective transformation coefficients.
    
    Args:
        source_coords (list): Original corner coordinates
        target_coords (list): Target corner coordinates
        
    Returns:
        list: Transformation coefficients
    """
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    
    A = np.matrix(matrix, dtype=float)
    B = np.array(source_coords).reshape(8)
    
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def main():
    """
    Main entry point for the program.
    """
    if len(sys.argv) < 2:
        print("\n❌ Usage: python DataAugmentation.py <directory_path>")
        print("\nExample:")
        print("   python DataAugmentation.py ./images")
        print("   python DataAugmentation.py /path/to/plant/images/directory")
        print("\nDirectory structure example:")
        print("   images/")
        print("   ├── Apple_Black_rot/")
        print("   │   ├── image (1).JPG")
        print("   │   ├── image (2).JPG")
        print("   ├── Apple_healthy/")
        print("   ├── Grape_Black_rot/")
        print("   └── Grape_healthy/")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    try:
        augmentor = DataAugmentor(directory_path)
        success = augmentor.run_augmentation()
        
        if success:
            print("✅ Augmentation process completed successfully!\n")
        else:
            print("⚠️  Augmentation process encountered issues.\n")
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except NotADirectoryError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
