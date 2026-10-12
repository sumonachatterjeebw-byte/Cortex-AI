from app.storage import storage_manager
from app.config import TEMP_UPLOADS_DIR

def test_temporary_session_creation_and_cleanup():
    # Create session
    session = storage_manager.create_session("Test_Session_Cleanup")
    assert session.session_dir.exists()
    
    # Write a dummy file in session
    test_file = session.session_dir / "dummy.txt"
    test_file.write_text("temporary data")
    assert test_file.exists()
    
    # Delete session
    deleted = storage_manager.delete_session(session.scan_id)
    assert deleted is True
    assert not session.session_dir.exists()
    
    # Clean all sweep
    storage_manager.clean_all_temp_files()
    assert len(list(TEMP_UPLOADS_DIR.iterdir())) == 0
