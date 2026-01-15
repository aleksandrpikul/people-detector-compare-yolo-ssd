from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ..utils import Detection

class BaseDetector(ABC):
    name: str

    @abstractmethod
    def detect_people(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Return detections for class 'person' only."""
        raise NotImplementedError
