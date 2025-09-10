"""YOLO label I/O utilities for reading and processing YOLO format labels."""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import tempfile
import shutil

CLS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}


def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Parse a single YOLO format line."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        return cls_id, x_center, y_center, width, height
    except (ValueError, IndexError):
        return None


def reconstruct_string(labels: List[Tuple[int, float]]) -> str:
    """Reconstruct string from YOLO labels by sorting by x_center."""
    if not labels:
        return ""
    
    sorted_labels = sorted(labels, key=lambda x: x[1])
    chars = []
    for cls_id, _ in sorted_labels:
        if cls_id in CLS_MAP:
            chars.append(CLS_MAP[cls_id])
    return ''.join(chars)


def read_yolo_file(content: str, compute_string: bool = True) -> Dict:
    """Read a single YOLO label file and return structured data."""
    lines = content.strip().split('\n')
    chars = []
    xs = []
    
    for line in lines:
        if not line.strip():
            continue
        parsed = parse_yolo_line(line)
        if parsed:
            cls_id, x_center, _, _, _ = parsed
            chars.append(cls_id)
            xs.append(x_center)
    
    result = {
        'chars': chars,
        'xs': xs
    }
    
    if compute_string:
        result['string'] = reconstruct_string(list(zip(chars, xs)))
    
    return result


def process_label_upload(uploaded_file, progress_callback=None) -> Tuple[Dict[str, Dict], List[str]]:
    """Process uploaded label files (zip or individual files).
    
    Returns:
        Tuple of (labels dict, error list)
    """
    labels = {}
    errors = []
    
    try:
        if uploaded_file.name.endswith('.zip'):
            file_content = uploaded_file.read()
            with zipfile.ZipFile(io.BytesIO(file_content)) as z:
                txt_files = [f for f in z.namelist() 
                           if f.endswith('.txt') and not f.startswith('__MACOSX')]
                
                total_files = len(txt_files)
                for i, filename in enumerate(txt_files):
                    try:
                        with z.open(filename) as f:
                            content = f.read().decode('utf-8')
                            img_name = Path(filename).stem
                            labels[img_name] = read_yolo_file(content, compute_string=False)
                        
                        if progress_callback and (i + 1) % 10 == 0:
                            progress_callback((i + 1) / total_files, f"Processed {i + 1}/{total_files} files")
                    except Exception as e:
                        errors.append(f"{filename}: {str(e)}")
                        
        elif uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            img_name = Path(uploaded_file.name).stem
            labels[img_name] = read_yolo_file(content, compute_string=False)
    except Exception as e:
        errors.append(f"Main error: {str(e)}")
    
    return labels, errors


def process_folder_upload(uploaded_files, progress_callback=None) -> Tuple[Dict[str, Dict], List[str]]:
    """Process multiple uploaded label files.
    
    Returns:
        Tuple of (labels dict, error list)
    """
    labels = {}
    errors = []
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                img_name = Path(uploaded_file.name).stem
                labels[img_name] = read_yolo_file(content, compute_string=False)
                
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback((i + 1) / total_files, f"Processed {i + 1}/{total_files} files")
        except Exception as e:
            errors.append(f"{uploaded_file.name}: {str(e)}")
    
    return labels, errors


def get_class_counts(labels: Dict[str, Dict]) -> Dict[int, int]:
    """Get per-class character counts from labels."""
    counts = {i: 0 for i in range(36)}
    
    for label_data in labels.values():
        for cls_id in label_data['chars']:
            if cls_id in counts:
                counts[cls_id] += 1
    
    return counts


def compute_slim_format(labels: Dict[str, Dict]) -> Dict[str, Dict]:
    """Convert to slim JSON format for efficient storage."""
    slim = {}
    for img_name, data in labels.items():
        slim[img_name] = {
            'chars': data['chars'],
            'xs': data['xs']
        }
    return slim