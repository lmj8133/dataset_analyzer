"""Metrics calculation for character recognition evaluation."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from rapidfuzz import distance as rf_distance


def levenshtein_with_alignment(s1: str, s2: str) -> Tuple[int, List[Tuple[int, int, str]]]:
    """
    Compute Levenshtein distance with alignment tracking.
    Returns (distance, alignment) where alignment is list of (i, j, operation).
    Operations: 'match', 'substitute', 'insert', 'delete'.
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            alignment.append((i-1, j-1, 'match'))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((i-1, j-1, 'substitute'))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((i-1, -1, 'delete'))
            i -= 1
        else:
            alignment.append((-1, j-1, 'insert'))
            j -= 1
    
    alignment.reverse()
    return int(dp[m][n]), alignment


def compute_plate_accuracy(gt_strings: List[str], pred_strings: List[str]) -> float:
    """Compute Plate Accuracy (license plate recognition rate)."""
    if not gt_strings:
        return 0.0
    
    exact_matches = sum(1 for gt, pred in zip(gt_strings, pred_strings) if gt == pred)
    return exact_matches / len(gt_strings)


def compute_char_accuracy(gt_strings: List[str], pred_strings: List[str]) -> Tuple[float, int, int]:
    """
    Compute character-level accuracy using Levenshtein distance.
    Returns (accuracy, total_edit_distance, total_gt_chars).
    """
    if not gt_strings:
        return 0.0, 0, 0
    
    total_edit_distance = 0
    total_gt_chars = 0
    
    for gt, pred in zip(gt_strings, pred_strings):
        ed = rf_distance.Levenshtein.distance(gt, pred)
        total_edit_distance += ed
        total_gt_chars += max(1, len(gt))
    
    accuracy = 1.0 - (total_edit_distance / total_gt_chars) if total_gt_chars > 0 else 0.0
    return accuracy, total_edit_distance, total_gt_chars


def compute_per_class_accuracy(
    gt_strings: List[str], 
    pred_strings: List[str],
    cls_map: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class character accuracy using alignment.
    Returns dict with character -> {'accuracy': float, 'support': int, 'correct': int}.
    """
    char_stats = {char: {'correct': 0, 'support': 0} for char in cls_map.values()}
    
    for gt, pred in zip(gt_strings, pred_strings):
        _, alignment = levenshtein_with_alignment(gt, pred)
        
        for i, j, op in alignment:
            if op == 'match':
                char = gt[i]
                if char in char_stats:
                    char_stats[char]['correct'] += 1
                    char_stats[char]['support'] += 1
            elif op == 'substitute':
                char = gt[i]
                if char in char_stats:
                    char_stats[char]['support'] += 1
            elif op == 'delete':
                char = gt[i]
                if char in char_stats:
                    char_stats[char]['support'] += 1
    
    for char in char_stats:
        support = char_stats[char]['support']
        correct = char_stats[char]['correct']
        char_stats[char]['accuracy'] = correct / support if support > 0 else 0.0
    
    return char_stats


def compute_delta_counts(
    current_counts: Dict[int, int], 
    previous_counts: Optional[Dict[int, int]] = None
) -> Dict[int, int]:
    """Compute delta counts between current and previous run."""
    if previous_counts is None:
        return current_counts.copy()
    
    delta = {}
    all_classes = set(current_counts.keys()) | set(previous_counts.keys())
    
    for cls in all_classes:
        curr = current_counts.get(cls, 0)
        prev = previous_counts.get(cls, 0)
        delta[cls] = curr - prev
    
    return delta


def evaluate_run(
    gt_labels: Dict[str, Dict],
    pred_labels: Dict[str, Dict],
    train_counts: Dict[int, int],
    cls_map: Dict[int, str]
) -> Dict:
    """
    Evaluate a single run against ground truth.
    Returns comprehensive metrics.
    """
    gt_strings = []
    pred_strings = []
    
    for img_name in gt_labels:
        if img_name in pred_labels:
            gt_str = gt_labels[img_name].get('string', '')
            pred_str = pred_labels[img_name].get('string', '')
            
            if not gt_str:
                from core.io_yolo import reconstruct_string
                gt_str = reconstruct_string(
                    list(zip(gt_labels[img_name]['chars'], gt_labels[img_name]['xs']))
                )
            if not pred_str:
                from core.io_yolo import reconstruct_string
                pred_str = reconstruct_string(
                    list(zip(pred_labels[img_name]['chars'], pred_labels[img_name]['xs']))
                )
            
            gt_strings.append(gt_str)
            pred_strings.append(pred_str)
    
    plate_accuracy = compute_plate_accuracy(gt_strings, pred_strings)
    char_acc, total_ed, total_chars = compute_char_accuracy(gt_strings, pred_strings)
    per_class_acc = compute_per_class_accuracy(gt_strings, pred_strings, cls_map)
    
    return {
        'plate_accuracy': plate_accuracy,
        'char_accuracy': char_acc,
        'total_edit_distance': total_ed,
        'total_gt_chars': total_chars,
        'n_plates': len(gt_strings),
        'per_class_accuracy': per_class_acc,
        'train_counts': train_counts
    }