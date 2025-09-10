"""Optimized YOLO label I/O utilities with batch processing."""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Generator
import gc

CLS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}


def process_zip_in_batches(
    uploaded_file,
    batch_size: int = 50,
    progress_callback: Optional[Callable] = None
) -> Tuple[Dict[str, Dict], List[str], Dict]:
    """
    Process ZIP file in batches to avoid memory issues.
    
    Returns:
        Tuple of (labels dict, error list, stats dict)
    """
    labels = {}
    errors = []
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'total_chars': 0,
        'avg_processing_time': 0
    }
    
    import time
    start_time = time.time()
    
    try:
        # Read ZIP file once
        file_content = uploaded_file.read()
        
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            # Get all txt files
            txt_files = [f for f in z.namelist() 
                        if f.endswith('.txt') and not f.startswith('__MACOSX')]
            
            stats['total_files'] = len(txt_files)
            
            # Process in batches
            for batch_start in range(0, len(txt_files), batch_size):
                batch_end = min(batch_start + batch_size, len(txt_files))
                batch_files = txt_files[batch_start:batch_end]
                
                for filename in batch_files:
                    try:
                        with z.open(filename) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            
                            # Quick parse without string reconstruction
                            chars, xs = parse_yolo_content_fast(content)
                            
                            if chars:  # Only store if has content
                                img_name = Path(filename).stem
                                labels[img_name] = {
                                    'chars': chars,
                                    'xs': xs
                                }
                                stats['total_chars'] += len(chars)
                                stats['processed_files'] += 1
                            
                    except Exception as e:
                        errors.append(f"{filename}: {str(e)}")
                        stats['failed_files'] += 1
                
                # Update progress
                if progress_callback:
                    progress = batch_end / len(txt_files)
                    elapsed = time.time() - start_time
                    rate = stats['processed_files'] / elapsed if elapsed > 0 else 0
                    eta = (len(txt_files) - batch_end) / rate if rate > 0 else 0
                    
                    message = (f"Processed {batch_end}/{len(txt_files)} files | "
                             f"Rate: {rate:.1f} files/sec | "
                             f"ETA: {eta:.1f}s")
                    progress_callback(progress, message)
                
                # Force garbage collection after each batch
                if batch_end % 200 == 0:
                    gc.collect()
    
    except Exception as e:
        errors.append(f"Critical error: {str(e)}")
    
    stats['avg_processing_time'] = (time.time() - start_time) / max(1, stats['processed_files'])
    
    return labels, errors, stats


def parse_yolo_content_fast(content: str) -> Tuple[List[int], List[float]]:
    """Fast parsing of YOLO content without string reconstruction."""
    chars = []
    xs = []
    
    for line in content.strip().split('\n'):
        if not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                
                # Validate class ID
                if 0 <= cls_id <= 35:
                    chars.append(cls_id)
                    xs.append(x_center)
            except (ValueError, IndexError):
                continue
    
    return chars, xs


def process_files_generator(
    uploaded_files,
    chunk_size: int = 20
) -> Generator[Tuple[str, Dict], None, None]:
    """
    Generator to process files in chunks, yielding results as they're ready.
    """
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.txt'):
            try:
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                chars, xs = parse_yolo_content_fast(content)
                
                if chars:
                    img_name = Path(uploaded_file.name).stem
                    yield img_name, {
                        'chars': chars,
                        'xs': xs
                    }
            except Exception:
                continue


def validate_yolo_format(content: str) -> Tuple[bool, str]:
    """
    Validate if content is in proper YOLO format.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    lines = content.strip().split('\n')
    
    if not lines:
        return False, "Empty file"
    
    for i, line in enumerate(lines[:5]):  # Check first 5 lines
        if not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) < 5:
            return False, f"Line {i+1}: Expected 5 values, got {len(parts)}"
        
        try:
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id > 35:
                return False, f"Line {i+1}: Class ID {cls_id} out of range (0-35)"
            
            for j in range(1, 5):
                val = float(parts[j])
                if j < 3 and (val < 0 or val > 1):  # x_center, y_center
                    return False, f"Line {i+1}: Coordinate {val} out of range (0-1)"
        except ValueError as e:
            return False, f"Line {i+1}: {str(e)}"
    
    return True, ""