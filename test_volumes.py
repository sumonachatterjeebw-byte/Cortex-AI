import numpy as np
import pytest
from app.metrics import compute_label_volumes

def test_volume_calculation_known_counts():
    # 2.0 mm isotropic grid -> 8.0 mm^3 per voxel
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    label_array = np.zeros((10, 10, 10), dtype=np.int16)
    
    # 100 voxels of Left Cerebral White Matter (label 2)
    label_array[:5, :5, :4] = 2
    
    # 500 voxels of Left Lateral Ventricle (label 4)
    label_array[5:10, :10, :10] = 4
    
    volumes = compute_label_volumes(label_array, affine)
    vol_dict = {v["label_id"]: v for v in volumes}
    
    # Label 2
    assert 2 in vol_dict
    assert vol_dict[2]["voxel_count"] == 100
    assert vol_dict[2]["volume_mm3"] == pytest.approx(800.0)
    assert vol_dict[2]["volume_ml"] == pytest.approx(0.8)
    assert vol_dict[2]["is_empty"] is False
    
    # Label 4
    assert 4 in vol_dict
    assert vol_dict[4]["voxel_count"] == 500
    assert vol_dict[4]["volume_mm3"] == pytest.approx(4000.0)
    assert vol_dict[4]["volume_ml"] == pytest.approx(4.0)

def test_empty_label_handling():
    # Challenge D3: Structure absent from output reported as 0 with explicit "empty label" warning
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    label_array = np.zeros((10, 10, 10), dtype=np.int16)
    
    # Expect label 17 (Left Hippocampus) even if 0 voxels present
    volumes = compute_label_volumes(label_array, affine, expected_labels=[17])
    hip_vol = next((v for v in volumes if v["label_id"] == 17), None)
    
    assert hip_vol is not None
    assert hip_vol["voxel_count"] == 0
    assert hip_vol["volume_mm3"] == 0.0
    assert hip_vol["volume_ml"] == 0.0
    assert hip_vol["is_empty"] is True
    assert any("Empty label" in w for w in hip_vol["warnings"])

def test_small_structure_warning():
    # Challenge D2: Voxel counts < 50 trigger warning
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    label_array = np.zeros((10, 10, 10), dtype=np.int16)
    label_array[0, 0, :10] = 18  # 10 voxels of Left Amygdala
    
    volumes = compute_label_volumes(label_array, affine)
    amy_vol = next((v for v in volumes if v["label_id"] == 18), None)
    
    assert amy_vol is not None
    assert amy_vol["voxel_count"] == 10
    assert any("Small structure" in w for w in amy_vol["warnings"])
