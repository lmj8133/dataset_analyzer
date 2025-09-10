"""Data persistence manager for saving and loading session data."""

import json
import gzip
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import streamlit as st


class DataManager:
    """Manages data persistence for the application."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session_file = self.data_dir / "session_data.json.gz"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.version = "1.0"
    
    def save_session(self, auto_save: bool = False) -> Tuple[bool, str]:
        """
        Save current session state to JSON file.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Prepare data for serialization
            data = {
                "version": self.version,
                "last_updated": datetime.now().isoformat(),
                "auto_save": auto_save,
                "gt_data": self._prepare_gt_data(),
                "runs": self._prepare_runs_data(),
                "settings": {
                    "auto_save_enabled": st.session_state.get("auto_save_enabled", True)
                }
            }
            
            # Create backup if file exists
            if self.session_file.exists():
                self._create_backup()
            
            # Save with compression
            with gzip.open(self.session_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Update last save time
            st.session_state.last_save_time = datetime.now()
            
            file_size = self.session_file.stat().st_size / 1024  # KB
            return True, f"Saved successfully ({file_size:.1f} KB)"
            
        except Exception as e:
            return False, f"Save failed: {str(e)}"
    
    def load_session(self) -> Tuple[bool, str]:
        """
        Load session data from JSON file.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.session_file.exists():
                return False, "No saved session found"
            
            # Load compressed data
            with gzip.open(self.session_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check version compatibility
            if data.get("version") != self.version:
                # Handle version migration if needed
                data = self._migrate_data(data)
            
            # Restore GT data
            if "gt_data" in data and data["gt_data"]:
                st.session_state.gt_labels = data["gt_data"].get("labels", {})
                st.session_state.gt_slim = data["gt_data"].get("labels", {})  # Same as labels
                
            # Restore runs
            if "runs" in data:
                runs = []
                for run_data in data["runs"]:
                    # Convert timestamp string back to datetime
                    if "timestamp" in run_data:
                        run_data["timestamp"] = datetime.fromisoformat(run_data["timestamp"])
                    runs.append(run_data)
                st.session_state.runs = runs
            
            # Restore settings
            if "settings" in data:
                st.session_state.auto_save_enabled = data["settings"].get("auto_save_enabled", True)
            
            last_updated = data.get("last_updated", "Unknown")
            gt_count = len(st.session_state.get("gt_labels", {}))
            run_count = len(st.session_state.get("runs", []))
            
            return True, f"Loaded: {gt_count} GT images, {run_count} runs (saved {last_updated})"
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}"
        except Exception as e:
            return False, f"Load failed: {str(e)}"
    
    def _prepare_gt_data(self) -> Optional[Dict]:
        """Prepare GT data for serialization."""
        if not hasattr(st.session_state, 'gt_labels') or not st.session_state.gt_labels:
            return {}
        
        # Calculate stats
        labels = st.session_state.gt_labels
        total_chars = sum(len(data['chars']) for data in labels.values())
        
        return {
            "labels": labels,
            "stats": {
                "total_images": len(labels),
                "total_chars": total_chars,
                "avg_plate_length": total_chars / len(labels) if labels else 0,
                "upload_time": st.session_state.get("gt_upload_time", datetime.now()).isoformat()
                if hasattr(st.session_state.get("gt_upload_time", datetime.now()), 'isoformat')
                else str(st.session_state.get("gt_upload_time", datetime.now()))
            }
        }
    
    def _prepare_runs_data(self) -> list:
        """Prepare runs data for serialization."""
        if not hasattr(st.session_state, 'runs'):
            return []
        
        runs = []
        for run in st.session_state.runs:
            run_copy = run.copy()
            # Convert datetime to string
            if "timestamp" in run_copy:
                run_copy["timestamp"] = run_copy["timestamp"].isoformat()
            runs.append(run_copy)
        
        return runs
    
    def _create_backup(self):
        """Create a backup of the current session file."""
        if self.session_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"session_backup_{timestamp}.json.gz"
            shutil.copy2(self.session_file, backup_file)
            
            # Keep only last 3 backups
            self._cleanup_old_backups()
    
    def _cleanup_old_backups(self, keep_count: int = 3):
        """Remove old backup files, keeping only the most recent ones."""
        backup_files = sorted(self.backup_dir.glob("session_backup_*.json.gz"))
        
        if len(backup_files) > keep_count:
            for old_file in backup_files[:-keep_count]:
                old_file.unlink()
    
    def _migrate_data(self, data: Dict) -> Dict:
        """Migrate data from old version to current version."""
        # Placeholder for version migration logic
        old_version = data.get("version", "0.0")
        
        # Add migration logic here as needed
        if old_version < self.version:
            # Perform necessary migrations
            pass
        
        data["version"] = self.version
        return data
    
    def export_data(self) -> Optional[bytes]:
        """
        Export current session data as downloadable bytes.
        
        Returns:
            Compressed JSON data as bytes
        """
        try:
            data = {
                "version": self.version,
                "exported_at": datetime.now().isoformat(),
                "gt_data": self._prepare_gt_data(),
                "runs": self._prepare_runs_data()
            }
            
            # Create in-memory gzip file
            import io
            buffer = io.BytesIO()
            with gzip.open(buffer, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            return buffer.getvalue()
            
        except Exception:
            return None
    
    def import_data(self, file_data: bytes) -> Tuple[bool, str]:
        """
        Import session data from uploaded file.
        
        Args:
            file_data: Uploaded file data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            import io
            
            # Try to decompress if gzipped
            try:
                with gzip.open(io.BytesIO(file_data), 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            except gzip.BadGzipFile:
                # Try as plain JSON
                data = json.loads(file_data.decode('utf-8'))
            
            # Validate structure
            if "gt_data" not in data and "runs" not in data:
                return False, "Invalid data format"
            
            # Create backup before importing
            if self.session_file.exists():
                self._create_backup()
            
            # Save imported data
            with gzip.open(self.session_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Load into session
            return self.load_session()
            
        except Exception as e:
            return False, f"Import failed: {str(e)}"
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current saved session."""
        if not self.session_file.exists():
            return {"exists": False}
        
        try:
            file_stat = self.session_file.stat()
            
            # Load metadata without full data
            with gzip.open(self.session_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "exists": True,
                "size_kb": file_stat.st_size / 1024,
                "last_modified": datetime.fromtimestamp(file_stat.st_mtime),
                "last_updated": data.get("last_updated"),
                "version": data.get("version"),
                "gt_count": len(data.get("gt_data", {}).get("labels", {})),
                "run_count": len(data.get("runs", []))
            }
        except Exception:
            return {"exists": True, "error": "Could not read file info"}
    
    def clear_all_data(self) -> bool:
        """Clear all saved data and backups."""
        try:
            # Remove session file
            if self.session_file.exists():
                self.session_file.unlink()
            
            # Remove all backups
            for backup_file in self.backup_dir.glob("*.json.gz"):
                backup_file.unlink()
            
            return True
        except Exception:
            return False


# Global instance
data_manager = DataManager()