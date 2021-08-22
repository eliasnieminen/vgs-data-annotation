import time
from pathlib import Path
from typing import Optional, Union

import cv2 as cv

from abc import ABC, abstractmethod

import numpy as np
from object_detection.utils import label_map_util
import od_utils


class Detector(ABC):
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
        if isinstance(img_obj, str):
            file = Path(img_obj).resolve()
            img = cv.imread(str(file))
        else:
            img = img_obj
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.blurriness = cv.Laplacian(img_gray, cv.CV_64F).var()

        del img, img_gray

        if self.blurriness > self.threshold:
            return True
        else:
            return False

    def get_blurriness(self):
        return self.blurriness


class ObjectDetector(Detector):
    def __init__(self, model_name: Optional[str] = "ssd_mobilenet_v1_coco_2017_11_17"):
        super(ObjectDetector, self).__init__()
        self.model_name = model_name

    def detect(self,
               img_obj: Union[str, np.ndarray, list],
               confidence_threshold: Optional[float] = 0.5):
        # labels_path = "data/coco/mscoco_label_map.pbtxt"
        # category_index \
        #     = label_map_util \
        #     .create_category_index_from_labelmap(labels_path,
        #                                          use_display_name=True)

        images_path = Path("img/").resolve()
        test_image_paths = sorted(images_path.glob("*.jpg"))

        detection_model = od_utils.load_model(self.model_name)

        print(detection_model.signatures["serving_default"].inputs)
        print(detection_model.signatures["serving_default"].output_dtypes)
        print(detection_model.signatures["serving_default"].output_shapes)

        start = time.time()

        num_detections = od_utils.get_num_objects(detection_model, img_obj)

        end = time.time()

        print(f"Images took {end - start} sec")

        return num_detections
