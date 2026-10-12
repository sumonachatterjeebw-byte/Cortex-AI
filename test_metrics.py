import numpy as np
import pytest
from app.metrics import compute_dice_score, compute_hd95_mm, evaluate_segmentation_against_reference

def test_dice_identical_masks():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:15, 5:15, 5:15] = 1
    dice = compute_dice_score(mask, mask)
    assert dice == pytest.approx(1.0)

def test_dice_disjoint_masks():
    m1 = np.zeros((20, 20, 20), dtype=np.uint8)
    m2 = np.zeros((20, 20, 20), dtype=np.uint8)
    m1[0:5, 0:5, 0:5] = 1
    m2[10:15, 10:15, 10:15] = 1
    dice = compute_dice_score(m1, m2)
    assert dice == pytest.approx(0.0)

def test_dice_both_empty():
    # Challenge D4: Both empty gives undefined (None)
    m1 = np.zeros((10, 10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10, 10), dtype=np.uint8)
    dice = compute_dice_score(m1, m2)
    assert dice is None

def test_dice_one_empty():
    m1 = np.zeros((10, 10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10, 10), dtype=np.uint8)
    m1[2:5, 2:5, 2:5] = 1
    dice = compute_dice_score(m1, m2)
    assert dice == pytest.approx(0.0)

def test_hd95_empty_mask():
    # Challenge D5: Empty mask returns None
    m1 = np.zeros((10, 10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10, 10), dtype=np.uint8)
    hd = compute_hd95_mm(m1, m2, (1.0, 1.0, 1.0))
    assert hd is None

def test_hd95_shifted_box():
    # Two identical 4x4x4 cubes shifted by 3mm along X with 1.0mm isotropic spacing
    m1 = np.zeros((20, 20, 20), dtype=np.uint8)
    m2 = np.zeros((20, 20, 20), dtype=np.uint8)
    m1[5:9, 5:9, 5:9] = 1
    m2[8:12, 5:9, 5:9] = 1
    hd = compute_hd95_mm(m1, m2, (1.0, 1.0, 1.0))
    assert hd is not None
    assert 2.0 <= hd <= 4.0

def test_evaluation_per_label_no_hiding():
    # Challenge D6: Ensure every label is listed in distribution
    pred = np.zeros((15, 15, 15), dtype=np.int16)
    ref = np.zeros((15, 15, 15), dtype=np.int16)
    # Good agreement for label 2
    pred[2:8, 2:8, 2:8] = 2
    ref[2:8, 2:8, 2:8] = 2
    # Poor agreement for label 4
    pred[9:14, 9:14, 9:14] = 4
    ref[11:15, 11:15, 11:15] = 4
    
    res = evaluate_segmentation_against_reference(pred, ref, (1.0, 1.0, 1.0))
    assert len(res["per_label_metrics"]) == 2
    labels = {r["label_id"]: r["dice"] for r in res["per_label_metrics"]}
    assert labels[2] == pytest.approx(1.0)
    assert labels[4] < 0.5
