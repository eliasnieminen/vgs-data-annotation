import time
from pathlib import Path
from typing import Optional, Union

import cv2 as cv

from abc import ABC, abstractmethod

import numpy as np
# from object_detection.utils import label_map_util
# import od_utils


class Detector(ABC):
    """Base class for a detector.

    """
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, img_path: str):
        pass


class BlurDetector(Detector):
    def __init__(self, threshold: Optional[float] = 10.0):
        super(BlurDetector, self).__init__()
        self.threshold = threshold
        self.blurriness = 0.0

    def detect(self, img_obj: Union[str, np.ndarray]):
        """Detects the blurriness from a given image path or np array.

        Args:
            img_obj: Either an np.array or image path.

        Returns: True, if the threshold is exceeded. False otherwise.

        """

        # Load the image.
        if isinstance(img_obj, str):
            file = Path(img_obj).resolve()
            img = cv.imread(str(file))
        else:
            img = img_obj

        # Compute the blurriness.
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.blurriness = cv.Laplacian(img_gray, cv.CV_64F).var()

        del img, img_gray

        if self.blurriness > self.threshold:
            return True
        else:
            return False

    def get_blurriness(self):
        """Return the blurriness value.

        Returns: Blurriness value as float.

        """
        return self.blurriness

