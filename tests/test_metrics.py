"""Unit tests for metrics calculations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.metrics import (
    levenshtein_with_alignment,
    compute_plate_accuracy,
    compute_char_accuracy,
    compute_per_class_accuracy,
    compute_delta_counts
)
from core.io_yolo import CLS_MAP


def test_levenshtein_with_alignment():
    """Test Levenshtein distance calculation with alignment."""
    s1 = "ABC123"
    s2 = "AB123"
    
    distance, alignment = levenshtein_with_alignment(s1, s2)
    
    assert distance == 1
    assert len(alignment) == 6
    
    operations = [op for _, _, op in alignment]
    assert 'delete' in operations
    assert operations.count('match') == 5


def test_levenshtein_substitution():
    """Test Levenshtein with substitution."""
    s1 = "ABC"
    s2 = "ADC"
    
    distance, alignment = levenshtein_with_alignment(s1, s2)
    
    assert distance == 1
    operations = [op for _, _, op in alignment]
    assert 'substitute' in operations
    assert operations.count('match') == 2


def test_compute_plate_accuracy():
    """Test Plate Accuracy calculation."""
    gt_strings = ["ABC123", "XYZ789", "DEF456"]
    pred_strings = ["ABC123", "XYZ789", "DEF457"]
    
    plate_accuracy = compute_plate_accuracy(gt_strings, pred_strings)
    assert plate_accuracy == pytest.approx(2/3)
    
    all_correct = ["ABC", "DEF", "GHI"]
    plate_accuracy_perfect = compute_plate_accuracy(all_correct, all_correct)
    assert plate_accuracy_perfect == 1.0
    
    plate_accuracy_empty = compute_plate_accuracy([], [])
    assert plate_accuracy_empty == 0.0


def test_compute_char_accuracy():
    """Test character-level accuracy calculation."""
    gt_strings = ["ABC", "DEF"]
    pred_strings = ["ABC", "DXF"]
    
    accuracy, total_ed, total_chars = compute_char_accuracy(gt_strings, pred_strings)
    
    assert total_ed == 1
    assert total_chars == 6
    assert accuracy == pytest.approx(5/6)
    
    perfect_acc, ed, chars = compute_char_accuracy(["ABC"], ["ABC"])
    assert perfect_acc == 1.0
    assert ed == 0
    assert chars == 3


def test_compute_per_class_accuracy():
    """Test per-class character accuracy calculation."""
    gt_strings = ["AAB", "ACC"]
    pred_strings = ["AAB", "ADC"]
    
    per_class = compute_per_class_accuracy(gt_strings, pred_strings, CLS_MAP)
    
    assert 'A' in per_class
    assert per_class['A']['support'] == 3
    assert per_class['A']['correct'] == 3
    assert per_class['A']['accuracy'] == 1.0
    
    assert 'C' in per_class
    assert per_class['C']['support'] == 2
    assert per_class['C']['correct'] == 1
    assert per_class['C']['accuracy'] == 0.5
    
    assert 'B' in per_class
    assert per_class['B']['support'] == 1
    assert per_class['B']['correct'] == 1
    assert per_class['B']['accuracy'] == 1.0


def test_compute_delta_counts():
    """Test delta counts calculation."""
    current = {0: 100, 1: 200, 2: 150}
    previous = {0: 80, 1: 200, 2: 160}
    
    delta = compute_delta_counts(current, previous)
    
    assert delta[0] == 20
    assert delta[1] == 0
    assert delta[2] == -10
    
    delta_first = compute_delta_counts(current, None)
    assert delta_first == current
    
    new_class = {0: 50, 3: 30}
    old_class = {0: 40, 2: 20}
    delta_mixed = compute_delta_counts(new_class, old_class)
    
    assert delta_mixed[0] == 10
    assert delta_mixed[2] == -20
    assert delta_mixed[3] == 30


def test_edge_cases():
    """Test edge cases for metrics."""
    empty_plate_accuracy = compute_plate_accuracy([], [])
    assert empty_plate_accuracy == 0.0
    
    single_char_acc, _, _ = compute_char_accuracy(["A"], ["B"])
    assert single_char_acc == 0.0
    
    distance, _ = levenshtein_with_alignment("", "ABC")
    assert distance == 3
    
    distance2, _ = levenshtein_with_alignment("ABC", "")
    assert distance2 == 3
    
    same_distance, alignment = levenshtein_with_alignment("ABC", "ABC")
    assert same_distance == 0
    assert all(op == 'match' for _, _, op in alignment)