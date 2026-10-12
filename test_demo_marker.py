import pytest
from app.reports import generate_csv_report, generate_html_report
from app.segmentation.demo_adapter import DemoSyntheticAdapter, DEMO_DATA_TAG

def test_demo_adapter_metadata_contains_marker():
    adapter = DemoSyntheticAdapter()
    guide = adapter.get_setup_guide()
    assert guide["is_demo"] is True
    assert "DEMO DATA" in guide["demo_tag"]

def test_demo_csv_export_contains_marker():
    sample_volumes = [
        {"label_id": 2, "name": "Left Cerebral White Matter", "voxel_count": 100, "volume_mm3": 800.0, "volume_ml": 0.8, "is_empty": False}
    ]
    csv_out = generate_csv_report("Demo_Test_Scan", sample_volumes, is_demo=True)
    assert "[DEMO DATA - NOT REAL AI INFERENCE]" in csv_out

def test_real_csv_export_does_not_contain_demo_marker():
    sample_volumes = [
        {"label_id": 2, "name": "Left Cerebral White Matter", "voxel_count": 100, "volume_mm3": 800.0, "volume_ml": 0.8, "is_empty": False}
    ]
    csv_out = generate_csv_report("Real_Test_Scan", sample_volumes, is_demo=False)
    assert "[DEMO DATA - NOT REAL AI INFERENCE]" not in csv_out

def test_demo_html_report_contains_marker():
    html_out = generate_html_report(
        alias="Demo_Test_Scan",
        metadata={"geometry": {"shape": [64, 64, 32], "zooms_mm": [1.0, 1.0, 1.0]}, "validation": {}, "iqm": {}},
        volumes=[],
        qc_report={},
        segmentation_job={"is_demo": True, "model_name": "Demo Pipeline"}
    )
    assert "DEMO DATA - SYNTHETIC DATA FOR UI VERIFICATION" in html_out
