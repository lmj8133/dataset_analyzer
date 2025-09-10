"""Generate test YOLO label files for testing."""

import zipfile
import random
import os
from pathlib import Path


def generate_yolo_label():
    """Generate a random YOLO label file content."""
    num_chars = random.randint(5, 10)  # Random plate length
    lines = []
    
    x_positions = sorted([random.random() for _ in range(num_chars)])
    
    for x in x_positions:
        # Random character class (0-35)
        cls_id = random.randint(0, 35)
        
        # Random y position
        y_center = 0.5 + random.uniform(-0.1, 0.1)
        
        # Random width and height
        width = random.uniform(0.05, 0.15)
        height = random.uniform(0.1, 0.2)
        
        lines.append(f"{cls_id} {x:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return "\n".join(lines)


def create_test_zip(num_files: int = 400, output_path: str = "test_labels.zip"):
    """Create a test ZIP file with YOLO labels."""
    print(f"Creating test ZIP with {num_files} files...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for i in range(num_files):
            filename = f"image_{i:04d}.txt"
            content = generate_yolo_label()
            z.writestr(filename, content)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_files} files...")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Created {output_path} ({file_size:.2f} MB)")
    return output_path


def create_test_files(num_files: int = 50, output_dir: str = "test_labels"):
    """Create individual test label files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating {num_files} test label files in {output_dir}/...")
    
    for i in range(num_files):
        filename = output_path / f"image_{i:04d}.txt"
        content = generate_yolo_label()
        filename.write_text(content)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_files} files...")
    
    print(f"✅ Created {num_files} files in {output_dir}/")
    return str(output_path)


def test_performance():
    """Test the performance with different file counts."""
    import time
    from core.io_yolo import process_label_upload
    
    test_sizes = [100, 400, 1000]
    
    print("\n🧪 Performance Testing\n" + "="*50)
    
    for num_files in test_sizes:
        print(f"\nTesting with {num_files} files:")
        
        # Create test ZIP
        zip_path = f"test_{num_files}.zip"
        create_test_zip(num_files, zip_path)
        
        # Simulate file upload
        with open(zip_path, 'rb') as f:
            file_content = f.read()
        
        # Create mock uploaded file
        class MockFile:
            def __init__(self, content, name):
                self.content = content
                self.name = name
            
            def read(self):
                return self.content
        
        mock_file = MockFile(file_content, zip_path)
        
        # Process and time it
        start_time = time.time()
        labels, errors = process_label_upload(mock_file)
        elapsed = time.time() - start_time
        
        # Calculate stats
        total_chars = sum(len(data['chars']) for data in labels.values())
        
        print(f"  ✅ Processed: {len(labels)} files")
        print(f"  ⏱️  Time: {elapsed:.2f} seconds")
        print(f"  📊 Rate: {len(labels)/elapsed:.1f} files/sec")
        print(f"  🔤 Total chars: {total_chars:,}")
        print(f"  ❌ Errors: {len(errors)}")
        
        # Clean up
        os.remove(zip_path)
    
    print("\n" + "="*50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_performance()
        elif sys.argv[1] == "zip":
            num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 400
            create_test_zip(num_files)
        elif sys.argv[1] == "files":
            num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            create_test_files(num_files)
    else:
        print("Usage:")
        print("  python generate_test_data.py test     # Run performance test")
        print("  python generate_test_data.py zip [N]  # Create ZIP with N files (default: 400)")
        print("  python generate_test_data.py files [N] # Create N individual files (default: 50)")
        print("\nCreating default test data...")
        create_test_zip(400)
        create_test_files(50)