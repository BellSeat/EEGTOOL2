"""
Diagnostic tool for ReplayManagerQt on macOS.
Run this to check if everything is set up correctly.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("Step 1: Checking Dependencies")
    print("="*60)
    
    required = {
        'PyQt5': 'PyQt5',
        'pyqtgraph': 'pyqtgraph', 
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True


def check_data_folder():
    """Check if data folder exists and has sessions."""
    print("\n" + "="*60)
    print("Step 2: Checking Data Folder")
    print("="*60)
    
    data_folder = Path("./data")
    
    if not data_folder.exists():
        print(f"✗ Data folder not found: {data_folder.absolute()}")
        print("Creating data folder...")
        data_folder.mkdir(parents=True, exist_ok=True)
        print("✓ Data folder created")
        return False
    else:
        print(f"✓ Data folder exists: {data_folder.absolute()}")
    
    # Check for metadata files
    metadata_files = list(data_folder.glob("*_metadata.json"))
    
    if not metadata_files:
        print("\n⚠️  No session metadata files found")
        print(f"Looking for: *_metadata.json in {data_folder}")
        
        # List what's in the folder
        all_files = list(data_folder.glob("*"))
        if all_files:
            print("\nFiles found in data folder:")
            for f in all_files[:10]:  # Show first 10
                print(f"  - {f.name}")
        else:
            print("\nData folder is empty")
        
        return False
    else:
        print(f"\n✓ Found {len(metadata_files)} session(s):")
        for f in metadata_files[:5]:  # Show first 5
            session_name = f.stem.replace('_metadata', '')
            print(f"  - {session_name}")
        
        if len(metadata_files) > 5:
            print(f"  ... and {len(metadata_files) - 5} more")
        
        return True


def check_datamanager():
    """Test if DataManager can load sessions."""
    print("\n" + "="*60)
    print("Step 3: Testing DataManager")
    print("="*60)
    
    try:
        from DataManager import DataManager
        print("✓ DataManager imported")
        
        dm = DataManager({"folder_path": "./data"})
        print("✓ DataManager initialized")
        
        sessions = dm.list_sessions()
        print(f"✓ DataManager.list_sessions() returned {len(sessions)} sessions")
        
        if sessions:
            print("\nSessions found:")
            for i, session in enumerate(sessions[:5]):
                print(f"  [{i+1}] {session}")
            return True
        else:
            print("\n⚠️  No sessions returned by DataManager")
            return False
            
    except Exception as e:
        print(f"✗ Error with DataManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pyqt():
    """Test if PyQt5 can create windows on macOS."""
    print("\n" + "="*60)
    print("Step 4: Testing PyQt5 on macOS")
    print("="*60)
    
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from PyQt5.QtCore import Qt
        
        print("✓ PyQt5 imported")
        
        # Check if running on macOS
        import platform
        if platform.system() == 'Darwin':
            print("✓ Running on macOS")
        
        # Create application
        app = QApplication(sys.argv)
        print("✓ QApplication created")
        
        # Try to show a simple message box
        print("\nAttempting to show a test dialog...")
        print("(Please check if a dialog appears on your screen)")
        
        msg = QMessageBox()
        msg.setWindowTitle("Test Dialog")
        msg.setText("If you can see this, PyQt5 is working!\n\n"
                   "Click OK to continue diagnostics.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        
        # Make sure window appears on top (macOS fix)
        msg.setWindowFlags(Qt.WindowStaysOnTopHint)
        msg.raise_()
        msg.activateWindow()
        
        result = msg.exec_()
        
        if result == QMessageBox.Ok:
            print("✓ Test dialog worked!")
            return True
        else:
            print("⚠️  Dialog may not have appeared")
            return False
            
    except Exception as e:
        print(f"✗ Error with PyQt5: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_dialog():
    """Test the actual SessionSelectionDialog."""
    print("\n" + "="*60)
    print("Step 5: Testing SessionSelectionDialog")
    print("="*60)
    
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
        from DataManager import DataManager
        
        # Import the dialog (need to extract from ReplayManagerQt.py)
        print("This step requires running the full ReplayManagerQt.py")
        print("If you reached here, try running: python ReplayManagerQt.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all diagnostic checks."""
    print("\n" + "="*60)
    print("ReplayManagerQt Diagnostic Tool (macOS)")
    print("="*60)
    
    results = []
    
    # Run checks
    results.append(("Dependencies", check_dependencies()))
    results.append(("Data Folder", check_data_folder()))
    results.append(("DataManager", check_datamanager()))
    results.append(("PyQt5", test_pyqt()))
    
    # Summary
    print("\n" + "="*60)
    print("Diagnostic Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All checks passed! ReplayManagerQt should work.")
        print("\nYou can now run: python ReplayManagerQt.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        
        # Provide specific help
        if not results[0][1]:
            print("\nTo fix missing dependencies:")
            print("  pip install PyQt5 pyqtgraph opencv-python numpy pandas")
        
        if not results[1][1]:
            print("\nTo create test data:")
            print("  python RecordingManager.py")
        
        if not results[2][1]:
            print("\nCheck if DataManager.py is in the same folder")
        
        if not results[3][1]:
            print("\nmacOS-specific PyQt5 issues:")
            print("  1. Check System Preferences → Security & Privacy")
            print("  2. Try: pip install --upgrade PyQt5")
            print("  3. Restart your terminal")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()