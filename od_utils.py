import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv


from pathlib import Path
from typing import Union, Optional
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_utils
from PIL import Image
from IPython.display import display
from google.protobuf import text_format
from util import format_2f


def load_model(model_name: str):
    """
    Loads the object detection model from Tensorflow's model zoo.
    :param model_name: The name of the model to be loaded.
    :return: The model that can be used to detect objects.
    """
    base_url = "http://download.tensorflow.org/models/object_detection/"
    model_file = f"{model_name}.tar.gz"
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_file,
                                        untar=True)

    model_dir = Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def run_inference(model, image):
    """

    :param model:
    :param image:
    :return:
    """
    image = np.asarray(image)
    tensor = tf.convert_to_tensor(image)
    input_tensor = tensor[tf.newaxis, ...]

    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    num_detections = output_dict.pop("num_detections")

    start_time = time.time()

    # with tf.Session() as sess:
    num_detections = int(num_detections.eval()[0])

    for key, value in output_dict.items():
        keyval_start_time = time.time()
        print(f"Processing {key}")
        output_dict[key] = np.array(value[0, :num_detections].eval())
        keyval_end_time = time.time()
        print(f"Loop took {keyval_end_time - keyval_start_time} sec")
    # ============
    end_time = time.time()
    print(f"Session took {end_time - start_time} sec")

    # output_dict = {key: value[0, :num_detections].numpy() \
    #                for key, value in output_dict.items()}

    output_dict["num_detections"] = num_detections

    num_threshold_detections \
        = output_dict["detection_scores"][output_dict["detection_scores"] > 0.5].shape[0]

    output_dict["detection_classes"] \
        = output_dict["detection_classes"].astype(np.int64)

    if "detection_masks" in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"], output_dict["detection_boxes"],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


def get_num_detections(model, image):
    """

    :param model:
    :param image:
    :return:
    """
    image = np.asarray(image)
    tensor = tf.convert_to_tensor(image)
    input_tensor = tensor[tf.newaxis, ...]

    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    num_detections = output_dict.pop("num_detections")

    start_time = time.time()

    # with tf.Session() as sess:
    num_detections = int(num_detections.numpy()[0])

    for key, value in output_dict.items():
        if key != "detection_scores":
            # print(f"Skipping {key}")
            continue

        keyval_start_time = time.time()
        # print(f"Processing {key}")
        output_dict[key] = value[0, :num_detections].numpy()
        keyval_end_time = time.time()
        # print(f"Loop took {keyval_end_time - keyval_start_time} sec")

    end_time = time.time()
    print(f"Session took {end_time - start_time} sec")

    det_scores = output_dict["detection_scores"]

    num_threshold_detections = det_scores[det_scores > 0.5].shape[0]

    return num_threshold_detections


def get_num_detections_from_dict(
        model,
        video_clips_and_frames: dict,
        num_objects_detected_threshold: Optional[int] = 1,
        obj_detection_confidence_threshold: Optional[float] = 0.3):
    """

    :param model:
    :param video_clips_and_frames:
    :param num_objects_detected_threshold:
    :param obj_detection_confidence_threshold:
    :return:
    """

    all_detections = {}

    for video in video_clips_and_frames.keys():

        clips_and_frames = video_clips_and_frames[video]

        clip_detections = {}

        for clip in clips_and_frames.keys():
            frames = clips_and_frames[clip]["frames"]
            frame_type = clips_and_frames[clip]["clip_info"]["frame_type"]

            if frame_type == "path":
                load_image = True
            else:
                load_image = False

            clip_detections[clip] = []
            for frame in frames:
                if load_image:
                    image = np.array(cv.imread(frame))
                else:
                    image = np.array(frame)
                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                tensor = tf.convert_to_tensor(image)
                tensor = tensor[tf.newaxis, ...]
                model_fn = model.signatures["serving_default"]
                output = model_fn(tensor)
                total_num_det = int(output.pop("num_detections").numpy()[0])
                for key, value in output.items():
                    if key not in ["detection_scores", "detection_classes"]:
                        continue
                    output[key] = value[0, :total_num_det].numpy()

                det_scores = output["detection_scores"]
                det_classes = np.ndarray.tolist(output["detection_classes"].astype(np.int16))
                label_map = get_label_map_dict()
                det_classes_names = []
                for cls in det_classes:
                    cls_name = label_map[cls]
                    det_classes_names.append(cls_name)

                num_threshold_detections \
                    = det_scores[det_scores >= obj_detection_confidence_threshold].shape[0]

                clip_detections[clip].append(num_threshold_detections)

                plt.imshow(image_rgb)
                count = 1
                for detection in det_scores[det_scores >= obj_detection_confidence_threshold]:
                    txt = f"{det_classes_names[int(detection)]} " \
                          f"{format_2f(det_scores[count - 1])}"
                    plt.text(100, count * 100, txt, color='r',
                             backgroundcolor='white')
                    count += 1
                det_id = time.time()
                plt.savefig(f"figures/detection/det{det_id}.png")
                plt.close()

        all_detections[video] = clip_detections

    print(all_detections)
    return all_detections


def get_label_map_dict():
    """

    :return:
    """
    labels_path = "data/coco/mscoco_label_map.pbtxt"

    f = open(labels_path, 'r')

    txt = f.read()

    spl = txt.split("id: ")

    label_map = {}
    count = 0
    for s in spl:
        if count == 0:
            count += 1
            continue

        label_index = int(s.split("\n")[0])
        disp_name = s.split("display_name: ")[1].split("\n")[0].replace('"', '')

        label_map[label_index] = disp_name

        count += 1

    return label_map


def get_num_objects(model, image):
    """

    :param model:
    :param image:
    :return:
    """
    is_list = False

    if isinstance(image, np.ndarray):
        image_np = image
    elif isinstance(image, list):
        is_list = True
        image_list = image
    else:
        image_np = np.array(Image.open(image))

    if is_list:
        dets = {}
        count = 0
        for img in image_list:
            num_detections = get_num_detections(model, img)
            dets[count] = num_detections
            count += 1
        num_detections = dets
    else:
        num_detections = get_num_detections(model, image_np)
    return num_detections


def show_inference(model, image_path):
    """

    :param model:
    :param image_path:
    :return:
    """
    file_name = Path(image_path).resolve().stem
    image_np = np.array(Image.open(image_path))
    num_detections = get_num_detections(model, image_np)
    print(f"Num detections for {file_name}: {num_detections}")

    # output_dict = run_inference(model, image_np)
    # vis_utils.visualize_boxes_and_labels_on_image_array(
    #     image=image_np,
    #     boxes=output_dict["detection_boxes"],
    #     classes=output_dict["detection_classes"],
    #     scores=output_dict["detection_scores"],
    #     category_index=category_index,
    #     instance_masks=output_dict.get("detection_masks_reframed", None),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    #
    # plt.imshow(image_np)
    # plt.savefig(f"figures/detection/{file_name}_detections.png")
    # display(Image.fromarray(image_np))
    return num_detections


