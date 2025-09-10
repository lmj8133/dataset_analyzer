# YOLO11 License Plate Character Recognition Dashboard

A minimal, fast dashboard for tracking license plate character recognition performance across YOLO11 training runs.

## Features

- **String-based metrics**: Plate accuracy and character accuracy using Levenshtein distance (no IoU/confidence)
- **In-memory storage**: All data kept in session state, no disk persistence
- **Real-time trends**: Interactive charts showing recognition rate evolution
- **Per-class analysis**: Track accuracy per character class
- **Training delta tracking**: Monitor training label changes between runs
- **Fast processing**: Optimized for ~10k GT characters

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run application
uv run streamlit run app.py
```

### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## Quick Start

1. **Upload GT Labels** (Ground Truth)
   - Navigate to "GT Upload" page
   - Upload your fixed test set labels (ZIP or multiple .txt files)
   - These labels remain constant across all runs

2. **Add Training Runs**
   - Navigate to "Add Run" page
   - For each training run, upload:
     - Training labels (for class counts)
     - Predictions on the fixed test set (for scoring)
   - Runs are auto-named with timestamps or provide custom names

3. **View Trends**
   - Navigate to "Trends" page (requires 2+ runs)
   - View overall recognition rate trends
   - Analyze per-class character accuracy
   - Monitor training label changes (delta counts)

## Data Format

### YOLO Label Format
Each `.txt` file contains one line per detected character:
```
class_id x_center y_center width height
```

- `class_id`: 0-9 for digits, 10-35 for letters A-Z
- Coordinates are normalized (0-1)
- No confidence scores in predictions

### Class Mapping
```python
0-9: Digits '0' to '9'
10-35: Letters 'A' to 'Z'
```

### Example Label File
```
10 0.5 0.3 0.1 0.15    # Letter A
11 0.6 0.3 0.1 0.15    # Letter B
12 0.7 0.3 0.1 0.15    # Letter C
0 0.8 0.3 0.1 0.15     # Digit 0
1 0.9 0.3 0.1 0.15     # Digit 1
```

## Metrics

### String-Level Metrics
- **Plate Accuracy**: Percentage of perfectly matched license plates
- **Character Accuracy**: 1 - (total edit distance / total GT characters)

### Character-Level Metrics
- **Per-Class Accuracy**: Accuracy for each character class (0-9, A-Z)
- Computed using Levenshtein alignment (no bounding box IoU)

### Training Metrics
- **Delta Counts**: Change in training label counts compared to previous run
- Helps track which characters are being added/removed in training data

## Project Structure

```
dataset_analyzer/
├── app.py                 # Main Streamlit application
├── core/
│   ├── io_yolo.py        # YOLO label I/O utilities
│   └── metrics.py        # Metric calculations
├── ui_pages/
│   ├── gt.py             # GT upload page
│   ├── add_run.py        # Run upload page
│   └── trends.py         # Trends visualization
├── tests/
│   └── test_metrics.py   # Unit tests
├── requirements.txt      # Dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Running Tests

```bash
# Using pytest
pytest tests/ -v

# Using uv
uv run pytest tests/ -v
```

## Performance

- Designed for ~10k total GT characters
- All processing in memory for speed
- String reconstruction via x_center sorting
- Efficient Levenshtein implementation via rapidfuzz

## Assumptions

1. Single license plate per image
2. Characters reconstructed by sorting on x_center (left-to-right)
3. GT labels remain fixed across all runs
4. No persistence - browser refresh clears all data

## License

MIT