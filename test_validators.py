import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from app.validators import validate_nifti_file, ValidationError

def test_reject_2d_file(tmp_path: Path):
    # Challenge A3: Reject 2D file
    data_2d = np.ones((64, 64), dtype=np.float32)
    img_2d = nib.Nifti1Image(data_2d, np.eye(4))
    file_path = tmp_path / "test_2d.nii.gz"
    nib.save(img_2d, str(file_path))
    
    with pytest.raises(ValidationError) as exc:
        validate_nifti_file(file_path)
    assert "Unsupported 2D" in str(exc.value)

def test_reject_4d_multivolume(tmp_path: Path):
    # Challenge A3: Reject 4D with >1 frames
    data_4d = np.ones((32, 32, 16, 5), dtype=np.float32)
    img_4d = nib.Nifti1Image(data_4d, np.eye(4))
    file_path = tmp_path / "test_4d.nii.gz"
    nib.save(img_4d, str(file_path))
    
    with pytest.raises(ValidationError) as exc:
        validate_nifti_file(file_path)
    assert "Unsupported 4D multi-volume" in str(exc.value)

def test_reject_all_zero_or_constant(tmp_path: Path):
    # Challenge A4: Reject all-zero
    data_zero = np.zeros((32, 32, 16), dtype=np.float32)
    img = nib.Nifti1Image(data_zero, np.eye(4))
    file_path = tmp_path / "test_zero.nii.gz"
    nib.save(img, str(file_path))
    
    with pytest.raises(ValidationError) as exc:
        validate_nifti_file(file_path)
    assert "completely uniform or empty" in str(exc.value)

def test_reject_corrupted_file(tmp_path: Path):
    # Challenge A7: Fake or corrupted file renamed to .nii
    fake_file = tmp_path / "corrupted.nii"
    fake_file.write_text("THIS IS NOT A NIFTI FILE HEADER CONTENT")
    
    with pytest.raises(ValidationError) as exc:
        validate_nifti_file(fake_file)
    assert "Invalid or corrupted NIfTI file" in str(exc.value)
