from pathlib import Path
from typing import List, Dict, Optional
import os
import glob
from dataclasses import dataclass
import time

@dataclass
class DocumentMetadata:
    file_name: str
    page_number: int
    chunk_index: int
    timestamp: float

class DocumentStore:
    def __init__(self, base_folder: str = "documents"):
        self.base_folder = base_folder
        self.preloaded_path = os.path.join(base_folder, "preloaded")
        self.uploaded_path = os.path.join(base_folder, "uploaded")
        
        # Create directories if they don't exist
        os.makedirs(self.preloaded_path, exist_ok=True)
        os.makedirs(self.uploaded_path, exist_ok=True)
    
    def get_all_pdfs(self) -> List[Path]:
        """Get all PDFs from both preloaded and uploaded folders"""
        preloaded = glob.glob(os.path.join(self.preloaded_path, "*.pdf"))
        uploaded = glob.glob(os.path.join(self.uploaded_path, "*.pdf"))
        return [Path(p) for p in preloaded + uploaded]
    
    def save_uploaded_pdf(self, file_data: bytes, filename: str) -> Path:
        """Save an uploaded PDF file"""
        file_path = os.path.join(self.uploaded_path, filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)
        return Path(file_path)
