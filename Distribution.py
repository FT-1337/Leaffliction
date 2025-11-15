#!/usr/bin/env python3
"""
Distribution.py
===============

A comprehensive plant disease classification and analysis tool.

Purpose:
- Scans a directory containing plant images organized by disease/health status
- Or analyzes a single plant image file
- Analyzes the distribution of different plant types and their conditions
- Generates visual reports: Pie charts and Bar charts

Input:
    - Directory path containing subdirectories of plant images
    - Or a single image file path
    - Each subdirectory represents a plant type/disease category
    - Supported formats: .jpg, .jpeg, .png, .gif, .bmp

Output:
    - Pie chart showing the distribution across all categories
    - Bar chart showing the count of images per category
    - Charts are saved to 'output/' directory

Example Usage:
    python Distribution.py ./images
    python Distribution.py ./images/Apple_Black_rot/image.jpg
    python Distribution.py /path/to/plant/images/directory

Project Structure:
    ├── images/                  # Input: Plant image directories
    │   ├── Apple_Black_rot/
    │   ├── Apple_healthy/
    │   ├── Grape_Black_rot/
    │   └── ...
    ├── output/                  # Output: Generated charts
    ├── Distribution.py          # This program
    ├── requirements.txt         # Dependencies
    └── venv/                    # Virtual environment
"""

import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


class PlantDataAnalyzer:
    """
    Analyzes plant image distribution and generates visualization charts.

    Attributes:
        data_directory (Path): Path to the images directory
        supported_formats (tuple): Tuple of supported image file extensions
        plant_data (dict): Dictionary storing image counts by category
    """

    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

    def __init__(self, directory_path):
        """
        Initialize the analyzer with a directory path or image file path.

        Args:
            directory_path (str): Path to the directory containing plant
                images or a single image file

        Raises:
            FileNotFoundError: If the path doesn't exist
            NotADirectoryError: If the path is not a directory and not a
                supported image file
            ValueError: If the file is not a supported image format
        """
        self.data_directory = Path(directory_path)

        if not self.data_directory.exists():
            raise FileNotFoundError(f"Path not found: {directory_path}")

        self.is_single_file = self.data_directory.is_file()

        if self.is_single_file:
            # Check if it's a supported image format
            suffix = self.data_directory.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                formats = ', '.join(self.SUPPORTED_FORMATS)
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    f"Supported formats: {formats}")

            # For single file, create plant_data with parent directory
            # as category
            category = self.data_directory.parent.name
            self.plant_data = {category: 1}
        else:
            # Directory case
            if not self.data_directory.is_dir():
                raise NotADirectoryError(
                    f"Path is not a directory or supported image file: "
                    f"{directory_path}")

            self.plant_data = defaultdict(int)

        self.output_directory = Path('./output')
        self.output_directory.mkdir(exist_ok=True)

    def scan_directory(self):
        """
        Scan the directory and count images in each subdirectory,
        or return single file data.

        Returns:
            dict: Dictionary with category names as keys and image
                counts as values
        """
        if self.is_single_file:
            print(f"\n🔍 Analyzing single image: {self.data_directory}")
            print("-" * 60)
            category = list(self.plant_data.keys())[0]
            count = self.plant_data[category]
            print(f"  📁 {category:<30} | 🖼️  {count} image")
            print("-" * 60)
            print(f"✅ Total images found: {count}\n")
            return self.plant_data

        print(f"\n🔍 Scanning directory: {self.data_directory}")
        print("-" * 60)

        subdirectories = [
            d for d in self.data_directory.iterdir() if d.is_dir()]

        if not subdirectories:
            print("⚠️  No subdirectories found in the specified path.")
            return self.plant_data

        for subdir in sorted(subdirectories):
            image_count = 0

            # Count images with supported formats
            for format_ext in self.SUPPORTED_FORMATS:
                image_count += len(list(subdir.glob(f'*{format_ext}')))
                image_count += len(list(subdir.glob(f'*{format_ext.upper()}')))

            if image_count > 0:
                self.plant_data[subdir.name] = image_count
                print(f"  📁 {subdir.name:<30} | 🖼️  "
                      f"{image_count} images")

        print("-" * 60)
        total_images = sum(self.plant_data.values())
        print(f"✅ Total images found: {total_images}\n")

        return self.plant_data

    def generate_pie_chart(self):
        """
        Generate a pie chart showing the distribution of images by category.
        """
        if not self.plant_data:
            print("⚠️  No data available. Run scan_directory() first.")
            return

        plt.figure(figsize=(10, 8))

        categories = list(self.plant_data.keys())
        counts = list(self.plant_data.values())

        # Create pie chart with enhanced styling
        colors = plt.cm.Set3(range(len(categories)))
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=categories,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 9}
        )

        # Enhance percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)

        plt.title('Plant Distribution by Category (Pie Chart)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = self.output_directory / 'distribution_pie_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Pie chart saved: "
              f"{output_path}")
        plt.close()

    def generate_bar_chart(self):
        """
        Generate a bar chart showing the count of images per category.
        """
        if not self.plant_data:
            print("⚠️  No data available. Run scan_directory() first.")
            return

        plt.figure(figsize=(12, 6))

        categories = list(self.plant_data.keys())
        counts = list(self.plant_data.values())

        # Create bar chart with enhanced styling
        bars = plt.bar(
            categories, counts,
            color=plt.cm.Set2(range(len(categories))),
            edgecolor='black', linewidth=1.2, alpha=0.8)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.xlabel('Plant Category', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
        plt.title('Plant Distribution by Category (Bar Chart)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        output_path = self.output_directory / 'distribution_bar_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Bar chart saved: "
              f"{output_path}")
        plt.close()

    def generate_summary_report(self):
        """
        Generate a summary report of the analysis.
        """
        if not self.plant_data:
            print("⚠️  No data available. Run scan_directory() first.")
            return

        # Create DataFrame
        df = pd.DataFrame(list(self.plant_data.items()),
                          columns=['Category', 'Count'])
        df['Percentage'] = (

            df['Count'] / df['Count'].sum() * 100

        ).round(2)
        df = df.sort_values('Count', ascending=False).reset_index(drop=True)

        # Display table
        print("\n" + "=" * 70)
        print("📈 ANALYSIS SUMMARY REPORT")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
        print(f"Total Categories: {len(df)}")
        print(f"Total Images: {df['Count'].sum()}")
        print(f"Average Images per Category: "
              f"{df['Count'].mean():.2f}")
        print("=" * 70 + "\n")

        # Save to CSV
        csv_path = self.output_directory / 'distribution_report.csv'
        df.to_csv(csv_path, index=False)
        print(f"📄 Report saved: "
              f"{csv_path}\n")

    def run_analysis(self):
        """
        Execute the complete analysis pipeline.
        """
        print("\n" + "=" * 60)
        print("🌿 PLANT DISEASE DISTRIBUTION ANALYZER")
        print("=" * 60)

        # Scan directory
        self.scan_directory()

        if not self.plant_data:
            print("❌ No images found. Please check your directory structure.")
            return False

        # Generate visualizations
        print("\n📊 Generating charts...")
        self.generate_pie_chart()
        self.generate_bar_chart()

        # Generate report
        self.generate_summary_report()

        print(f"✨ All outputs saved to: "
              f"{self.output_directory.absolute()}")
        return True


def main():
    """
    Main entry point for the program.
    """
    if len(sys.argv) < 2:
        print("\n❌ Usage: python Distribution.py "
              "<directory_path_or_image_file>")
        print("\nExamples:")
        print("   python Distribution.py ./images")
        print("   python Distribution.py "
              "./images/Apple_Black_rot/image.jpg")
        print("   python Distribution.py "
              "/path/to/plant/images/directory")
        print("\nDirectory structure example:")
        print("   images/")
        print("   ├── Apple_Black_rot/")
        print("   ├── Apple_healthy/")
        print("   ├── Grape_Black_rot/")
        print("   └── Grape_healthy/")
        sys.exit(1)

    directory_path = sys.argv[1]

    try:
        analyzer = PlantDataAnalyzer(directory_path)
        analyzer.run_analysis()
        print("\n✅ Analysis completed successfully!\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except NotADirectoryError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
