import cv2
from pathlib import Path

class Video:
    def __init__(self, path):
        self.path = Path(path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        
        # Open briefly just to read metadata
        cap = cv2.VideoCapture(str(self.path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def duration_seconds(self):
        return self.frame_count / self.fps

    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.path))
        return self._cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cap.release()
        return False  # don't suppress exceptions