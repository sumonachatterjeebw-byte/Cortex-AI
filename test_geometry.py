import numpy as np
import pytest
from app.geometry import compute_voxel_volume_mm3, validate_overlay_geometry

def test_voxel_volume_isotropic():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    vol = compute_voxel_volume_mm3(affine)
    assert vol == pytest.approx(1.0, abs=1e-5)

def test_voxel_volume_anisotropic():
    affine = np.diag([-1.5, 2.0, 3.0, 1.0])
    vol = compute_voxel_volume_mm3(affine)
    assert vol == pytest.approx(9.0, abs=1e-5)

def test_voxel_volume_sheared_oblique():
    # Affine with off-diagonal shear components
    affine = np.array([
        [1.0, 0.2, 0.0, 0.0],
        [0.0, 2.0, 0.1, 0.0],
        [0.1, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    det = np.abs(np.linalg.det(affine[:3, :3]))
    vol = compute_voxel_volume_mm3(affine)
    assert vol == pytest.approx(det, abs=1e-5)

def test_validate_overlay_geometry_matching():
    shape = (64, 64, 32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    valid, msg = validate_overlay_geometry(shape, affine, shape, affine)
    assert valid is True
    assert "aligned" in msg

def test_validate_overlay_geometry_mismatched_shape():
    shape1 = (64, 64, 32)
    shape2 = (64, 64, 33)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    valid, msg = validate_overlay_geometry(shape1, affine, shape2, affine)
    assert valid is False
    assert "Shape mismatch" in msg

def test_validate_overlay_geometry_mismatched_affine():
    shape = (64, 64, 32)
    affine1 = np.diag([1.0, 1.0, 1.0, 1.0])
    affine2 = np.diag([1.5, 1.0, 1.0, 1.0])
    valid, msg = validate_overlay_geometry(shape, affine1, shape, affine2)
    assert valid is False
    assert "Affine" in msg
