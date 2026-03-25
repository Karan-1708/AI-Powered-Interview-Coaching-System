import pytest
import os
import json
from src.utils.history import HistoryManager, HISTORY_FILE
from src.utils.file_manager import FileManager

@pytest.fixture
def clean_history():
    """Fixture to ensure a clean history file for tests."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    yield
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def test_save_and_load_history(clean_history):
    """Test saving and then loading session history."""
    HistoryManager.save_session(150, 2, "Confident", "Standard Interview")
    
    history = HistoryManager.load_history()
    assert len(history) == 1
    assert history[0]["wpm"] == 150
    assert history[0]["fillers"] == 2
    assert history[0]["tone"] == "Confident"
    assert history[0]["mode"] == "Standard Interview"

def test_clear_history(clean_history):
    """Test clearing the session history."""
    HistoryManager.save_session(150, 2, "Confident", "Standard Interview")
    assert os.path.exists(HISTORY_FILE)
    
    HistoryManager.clear_history()
    assert not os.path.exists(HISTORY_FILE)
    assert HistoryManager.load_history() == []

def test_file_manager_init():
    """Test directory initialization."""
    FileManager.initialize_directories()
    assert os.path.exists(FileManager.TEMP_DIR)
    assert os.path.exists(FileManager.LOG_DIR)

def test_file_manager_cleanup():
    """Test that cleanup removes dummy files but ignores the active log."""
    FileManager.initialize_directories()
    
    # Create dummy files
    dummy_wav = os.path.join(FileManager.TEMP_DIR, "test_dummy.wav")
    with open(dummy_wav, "w") as f:
        f.write("dummy")
        
    dummy_log = os.path.join(FileManager.LOG_DIR, "old_debug.log")
    with open(dummy_log, "w") as f:
        f.write("dummy log")
        
    active_log = os.path.join(FileManager.LOG_DIR, "app_debug.log")
    with open(active_log, "w") as f:
        f.write("active log")
        
    FileManager.cleanup_all_data()
    
    assert not os.path.exists(dummy_wav)
    assert not os.path.exists(dummy_log)
    assert os.path.exists(active_log) # Should be skipped
