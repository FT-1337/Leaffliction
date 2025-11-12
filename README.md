# 🌿 Plant Disease Distribution Analyzer

A Python-based analysis tool for studying plant disease distribution across image datasets.

## 📋 Project Overview

This project analyzes a directory structure containing plant images organized by disease/health status and generates statistical visualizations.

### What It Does

- **Scans** a directory containing subdirectories of plant images
- **Counts** images in each category (plant type/disease)
- **Analyzes** the distribution across all categories
- **Generates** professional visualizations:
  - 📊 **Pie Chart** - Shows percentage distribution
  - 📊 **Bar Chart** - Shows image count per category
  - 📄 **CSV Report** - Detailed statistics

## 📁 Project Structure

```
leaffliction/
├── Distribution.py              # Main program (executable)
├── requirements.txt             # Python dependencies
├── setup.sh                     # Setup script for virtual environment
├── venv/                        # Virtual environment (created after setup)
├── images/                      # Input: Plant image directories
│   ├── Apple_Black_rot/
│   ├── Apple_healthy/
│   ├── Apple_rust/
│   ├── Apple_scab/
│   ├── Grape_Black_rot/
│   ├── Grape_Esca/
│   ├── Grape_healthy/
│   └── Grape_spot/
├── output/                      # Output: Generated charts and reports
│   ├── distribution_pie_chart.png
│   ├── distribution_bar_chart.png
│   └── distribution_report.csv
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Set Up Virtual Environment

```bash
# Navigate to project directory
cd /Users/mac/Documents/42/leaffliction

# Run the setup script (one-time setup)
bash setup.sh
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### 3. Run the Program

```bash
# Using the default images directory
python Distribution.py ./images

# Or specify a custom directory
python Distribution.py /path/to/images/directory
```

## 📊 Output Files

The program generates three output files in the `output/` directory:

1. **distribution_pie_chart.png** - Visual pie chart showing percentage distribution
2. **distribution_bar_chart.png** - Visual bar chart showing image counts
3. **distribution_report.csv** - Spreadsheet with detailed statistics

## 📋 Input Requirements

- A directory containing subdirectories (one per plant category)
- Each subdirectory must contain image files in supported formats:
  - `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`

### Example Structure:
```
images/
├── Apple_Black_rot/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple_healthy/
│   ├── image1.jpg
│   └── ...
└── Grape_Black_rot/
    └── ...
```

## 🛠️ Dependencies

All dependencies are listed in `requirements.txt`:

- **matplotlib** - Data visualization and chart generation
- **pandas** - Data analysis and CSV export
- **numpy** - Numerical computing
- **Pillow** - Image processing support

Install them with: `pip install -r requirements.txt`

## 💻 Program Features

### PlantDataAnalyzer Class

The main `PlantDataAnalyzer` class provides:

- **scan_directory()** - Counts images in each subdirectory
- **generate_pie_chart()** - Creates percentage distribution visualization
- **generate_bar_chart()** - Creates count-based visualization
- **generate_summary_report()** - Creates statistical CSV report
- **run_analysis()** - Executes complete analysis pipeline

## 📝 Usage Examples

### Basic Usage
```bash
python Distribution.py ./images
```

### Custom Directory
```bash
python Distribution.py /Users/mac/Documents/plant_data
```

### With Custom Output
The program automatically creates an `output/` directory and saves all charts there.

## 🔧 Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```

## 📖 Code Documentation

The code includes:
- Comprehensive docstrings for all classes and methods
- Type hints for better code clarity
- Inline comments explaining complex logic
- Clear error handling and user feedback

## ✅ Requirements

- Python 3.7+
- macOS/Linux/Windows
- Terminal access

## 📊 Example Output

When you run the program, you'll see:

```
============================================================
🌿 PLANT DISEASE DISTRIBUTION ANALYZER
============================================================

🔍 Scanning directory: ./images
------------------------------------------------------------
  📁 Apple_Black_rot                   | 🖼️  120 images
  📁 Apple_healthy                     | 🖼️  200 images
  📁 Apple_rust                        | 🖼️  80 images
  📁 Apple_scab                        | 🖼️  150 images
  📁 Grape_Black_rot                   | 🖼️  95 images
  📁 Grape_Esca                        | 🖼️  110 images
  📁 Grape_healthy                     | 🖼️  180 images
  📁 Grape_spot                        | 🖼️  65 images
------------------------------------------------------------
✅ Total images found: 1000

📊 Generating charts...
📊 Pie chart saved: ./output/distribution_pie_chart.png
📊 Bar chart saved: ./output/distribution_bar_chart.png

======================================================================
📈 ANALYSIS SUMMARY REPORT
======================================================================
        Category  Count  Percentage
 Apple_healthy    200      20.00
 Apple_scab       150      15.00
 Apple_Black_rot  120      12.00
...
======================================================================
```

## 🐛 Troubleshooting

### Virtual Environment Won't Activate
```bash
# Ensure you're in the project directory
cd /Users/mac/Documents/42/leaffliction
source venv/bin/activate
```

### Module Not Found Error
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Directory Not Found
```bash
# Verify the path exists
ls ./images
```

## 📚 Educational Value

This project demonstrates:
- File system navigation and directory scanning
- Data collection and analysis
- Data visualization with matplotlib
- Object-oriented programming in Python
- Data export to CSV format
- Error handling and user feedback
- Documentation best practices

---

**Project:** 42 School - leaffliction  
**Purpose:** Plant Disease Distribution Analysis  
**Status:** ✅ Complete
