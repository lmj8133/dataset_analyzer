"""Test script for data persistence functionality."""

import json
import gzip
from pathlib import Path
from datetime import datetime
from core.data_manager import DataManager


def test_persistence():
    """Test the data persistence functionality."""
    print("\n🧪 Testing Data Persistence\n" + "="*50)
    
    # Initialize data manager
    dm = DataManager("test_data")
    
    # Test 1: Save and load simple data
    print("\n1. Testing basic save/load...")
    
    # Create mock session state
    class MockSessionState:
        def __init__(self):
            self.gt_labels = {
                "img_001": {"chars": [10, 11, 12, 0, 1, 2], "xs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                "img_002": {"chars": [20, 21, 22], "xs": [0.2, 0.4, 0.6]}
            }
            self.gt_slim = self.gt_labels
            self.runs = [
                {
                    "name": "test_run_1",
                    "description": "Test run 1",
                    "timestamp": datetime.now(),
                    "metrics": {
                        "emr": 0.85,
                        "char_accuracy": 0.92,
                        "total_edit_distance": 50,
                        "total_gt_chars": 500,
                        "n_images": 100,
                        "per_class_accuracy": {}
                    },
                    "train_counts": {i: 100 for i in range(36)}
                }
            ]
            self.auto_save_enabled = True
            self.last_save_time = None
            self.gt_upload_time = datetime.now()
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # Mock streamlit session state
    import sys
    import types
    
    mock_st = types.ModuleType('streamlit')
    mock_st.session_state = MockSessionState()
    sys.modules['streamlit'] = mock_st
    
    # Save session
    success, message = dm.save_session()
    print(f"  Save: {'✅' if success else '❌'} {message}")
    
    # Check file exists
    assert dm.session_file.exists(), "Session file not created"
    file_size = dm.session_file.stat().st_size / 1024
    print(f"  File size: {file_size:.2f} KB")
    
    # Load and verify
    with gzip.open(dm.session_file, 'rt') as f:
        loaded_data = json.load(f)
    
    assert "version" in loaded_data
    assert "gt_data" in loaded_data
    assert "runs" in loaded_data
    
    # Check gt_data
    if loaded_data["gt_data"] and "labels" in loaded_data["gt_data"]:
        assert len(loaded_data["gt_data"]["labels"]) == 2
        print(f"  GT labels: {len(loaded_data['gt_data']['labels'])}")
    assert len(loaded_data["runs"]) == 1
    print(f"  Runs: {len(loaded_data['runs'])}")
    print("  ✅ Data structure verified")
    
    # Test 2: Session info
    print("\n2. Testing session info...")
    info = dm.get_session_info()
    print(f"  Exists: {info.get('exists')}")
    print(f"  GT count: {info.get('gt_count')}")
    print(f"  Run count: {info.get('run_count')}")
    print(f"  Size: {info.get('size_kb', 0):.2f} KB")
    
    # Test 3: Backup creation
    print("\n3. Testing backup creation...")
    
    # Save again to trigger backup
    dm.save_session()
    backup_files = list(dm.backup_dir.glob("*.json.gz"))
    print(f"  Backup files: {len(backup_files)}")
    assert len(backup_files) >= 1, "No backup created"
    
    # Test 4: Export/Import
    print("\n4. Testing export/import...")
    
    export_data = dm.export_data()
    assert export_data is not None, "Export failed"
    print(f"  Export size: {len(export_data) / 1024:.2f} KB")
    
    # Test import
    success, message = dm.import_data(export_data)
    print(f"  Import: {'✅' if success else '❌'} {message}")
    
    # Test 5: Clear data
    print("\n5. Testing clear all data...")
    
    success = dm.clear_all_data()
    assert success, "Clear data failed"
    assert not dm.session_file.exists(), "Session file not deleted"
    print("  ✅ All data cleared")
    
    # Clean up test directory
    import shutil
    shutil.rmtree("test_data", ignore_errors=True)
    
    print("\n" + "="*50)
    print("✅ All tests passed!")


def test_compression_ratio():
    """Test compression efficiency for different data sizes."""
    print("\n📊 Testing Compression Ratios\n" + "="*50)
    
    dm = DataManager("test_compression")
    
    sizes = [10, 50, 100, 400, 1000]
    
    for num_images in sizes:
        # Create mock data
        class MockState:
            def __init__(self, n):
                self.gt_labels = {
                    f"img_{i:04d}": {
                        "chars": [j % 36 for j in range(5, 10)],
                        "xs": [0.1 * j for j in range(5)]
                    }
                    for i in range(n)
                }
                self.gt_slim = self.gt_labels
                self.runs = []
                self.auto_save_enabled = True
                self.gt_upload_time = datetime.now()
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        import sys
        import types
        mock_st = types.ModuleType('streamlit')
        mock_st.session_state = MockState(num_images)
        sys.modules['streamlit'] = mock_st
        
        # Save and measure
        dm.save_session()
        
        compressed_size = dm.session_file.stat().st_size / 1024
        
        # Calculate uncompressed size
        with gzip.open(dm.session_file, 'rt') as f:
            json_str = f.read()
        uncompressed_size = len(json_str.encode()) / 1024
        
        ratio = (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0
        
        print(f"\n  {num_images} images:")
        print(f"    Uncompressed: {uncompressed_size:.2f} KB")
        print(f"    Compressed: {compressed_size:.2f} KB")
        print(f"    Compression: {ratio:.1f}%")
    
    # Clean up
    import shutil
    shutil.rmtree("test_compression", ignore_errors=True)
    
    print("\n" + "="*50)


if __name__ == "__main__":
    test_persistence()
    test_compression_ratio()