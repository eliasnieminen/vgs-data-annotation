import cv2 as cv
import math

import librosa
import numpy as np
import od_utils
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import util
import pickle

from env import ProjectEnvironment
from pathlib import Path
from typing import Optional, List, Dict, Union
from video_metadata import VideoMetadata
from image_quality import BlurDetector

env = ProjectEnvironment()


def get_dataset_video_paths(target_dataset: str,
                            target_split: str,
                            allowed_video_suffixes: List):
    """Returns the paths to all videos in the given target dataset.

    Args:
        target_dataset:
        target_split:
        allowed_video_suffixes:

    Returns:

    """

    dataset_save_path = env[f"{target_dataset}_save_path"] \
        if target_split == "train" \
        else env[f"{target_dataset}_save_path_{target_split}"]

    video_paths = []
    for file in Path(dataset_save_path).resolve().iterdir():
        if file.suffix in allowed_video_suffixes:
            video_paths.append(str(file))

    return video_paths


def get_yt_id_from_video_path(video_path):
    """

    Args:
        video_path:

    Returns:

    """
    p = Path(video_path).resolve()
    stem = p.stem.replace(".mp4", "")
    spl = stem.split("_")
    yt_id = "_".join(spl[1:])
    return yt_id


def get_task_id_from_video_path(video_path: str):
    """

    Args:
        video_path:

    Returns:

    """
    p = Path(video_path).resolve()
    stem = p.stem.replace(".mp4", "")
    spl = stem.split("_")
    task_id = spl[0]
    return task_id


def get_video_metadata(video_path: str, original_dataset: str):
    """

    Args:
        video_path:
        original_dataset:

    Returns:

    """
    yt_id = get_yt_id_from_video_path(video_path)
    task_id = get_task_id_from_video_path(video_path)

    cap = cv.VideoCapture(video_path)

    while not cap.isOpened():
        cap = cv.VideoCapture(video_path)
        cv.waitKey(1000)
        print("Waiting for the header...")

    fps = float("{:.2f}".format(cap.get(cv.CAP_PROP_FPS)))
    dur = math.floor(cap.get(cv.CAP_PROP_FRAME_COUNT) / fps)

    vd = VideoMetadata(
        dur=dur,
        fps=fps,
        metadata={
            "filename": video_path,
             "yt_id": yt_id,
             "task_id": task_id,
             "original_dataset": original_dataset
        })
    return vd


def get_random_clips_as_list(n_clips: int,
                             clip_length: float,
                             video_metadata: VideoMetadata,
                             sort_clips: Optional[bool] = True):
    """

    Args:
        n_clips:
        clip_length:
        video_metadata:
        sort_clips:

    Returns:

    """
    fps = video_metadata.fps

    starts = np.random.rand(n_clips) * (video_metadata.duration - clip_length)
    starts_as_frames = np.round(starts * fps).astype(np.int64)

    if sort_clips:
        starts = np.sort(starts)
        starts_as_frames = np.sort(starts_as_frames)
    clips = []
    clips_as_frames = []
    for s in starts:
        clips.append((s, s + clip_length))
    for f in starts_as_frames:
        clips_as_frames.append((f, f + np.round(clip_length * fps).astype(np.int64)))

    return clips, clips_as_frames


def get_annotated_clips_as_list(yt_id: str,
                                target_dataset: str,
                                target_split: str):
    """

    Args:
        yt_id:
        target_dataset:
        target_split:

    Returns:

    """
    annotations_path = f"{env['clip_annotations_save_path']}" \
                       f"{target_dataset}/{target_split}/"

    clips = []
    clips_as_frames = []
    clip_infos = []
    for file in Path(annotations_path).resolve().iterdir():
        if yt_id in file.name and file.suffix == ".pickle":
            with open(str(file), 'rb') as clip_file:
                clip_info = pickle.load(clip_file)
                segment = clip_info["segment"]

                start_t = segment["segment_t"][0]
                end_t = segment["segment_t"][1]
                start_f = segment["segment_f"][0]
                end_f = segment["segment_f"][1]

                clips.append((start_t, end_t))
                clips_as_frames.append((start_f, end_f))
                clip_infos.append(clip_info)

    if len(clips) == 0:
        raise ValueError("The requested video was not found in this dataset.")

    return clips, clips_as_frames, clip_infos


def select_sharpest_frames(video_path: str,
                           start_f: int,
                           end_f: int,
                           video_metadata: VideoMetadata,
                           blur_threshold: Optional[float] = 10.0,
                           plotting: Optional[bool] = False,
                           return_type: Optional[str] = "numpy",
                           clip_id: Optional[int] = 0,
                           save_dir: Optional[Union[str, None]] = None):
    """

    Args:
        video_path:
        start_f:
        end_f:
        video_metadata:
        blur_threshold:
        plotting:
        return_type:
        clip_id:
        save_dir:

    Returns:

    """

    cap = cv.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv.VideoCapture(video_path)
        print("Waiting...")

    cap.set(cv.CAP_PROP_POS_FRAMES, start_f)
    pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)

    blur_detector = BlurDetector(threshold=blur_threshold)
    # object_detector = ObjectDetector()

    latest_second_frames = []
    best_frames = []
    fps = int(np.round(video_metadata.fps))

    # The frames get stuck from time to time (open-cv issue) so there needs
    # to be a way to see if the waiting time is too long for a frame. The
    # patience variable defines the maximum waiting time for each frame.
    wait_cumul = 0  # ms
    patience = 3000  # ms

    count = pos_frame
    frame_count = 0

    while True:
        flag, frame = cap.read()

        if flag:  # If reading was successful.
            # Reset the cumulative waiting time.
            wait_cumul = 0
            # Add this frame to the latest frames.
            latest_second_frames.append(frame)
            # Get the current position of the frames
            pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
            count += 1
            if count % fps == 0:
                # Process latest second
                blurs = []
                for fr in latest_second_frames:
                    # Detect blurriness for each frame with the BlurDetector
                    blur_detector.detect(fr)
                    blurriness = blur_detector.get_blurriness()

                    # Append each blurriness to a list
                    blurs.append(blurriness)

                # Find out the maximum value's index in the blur list, which
                # means the least blurred image.
                blurs = np.array(blurs)
                min_blur_index = np.argmax(blurs)
                best_frame = latest_second_frames[min_blur_index]

                # Add the best frame. Depending of the return type, the frame
                # will be in the form of a path or a numpy array.
                if return_type == "numpy":
                    best_frames.append(best_frame)
                elif return_type == "path":
                    best_frames.append(util.save_frame(
                        save_dir=save_dir,
                        frame=best_frame,
                        yt_id=video_metadata.metadata["yt_id"],
                        clip_id=clip_id,
                        frame_id=frame_count))

                if plotting:
                    plt.imshow(best_frame)
                    plt.show()
                    print("Plot show")

                # Re-initialize the list for the next second.
                print(f"Sec {pos_frame} count: {count}, start_f: {start_f} end_f: {end_f} done")
                latest_second_frames = []
                frame_count += 1
        else:
            print(f"wait_cumul: {wait_cumul}")
            if wait_cumul >= patience:
                print("Breaking due to stuck frame...")
                break
            cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("Waiting for frame...")
            wait_cumul += 250
            cv.waitKey(250)

        if count == end_f:
            break

    # After determining the best frame for each second, return the results.
    return best_frames


def detect_objects(clips_and_frames: dict,
                   model,
                   num_detected_objects_threshold: Optional[int] = 1,
                   obj_detection_confidence_threshold: Optional[float] = 0.3):
    detections = od_utils.get_num_detections_from_dict(
        model,
        clips_and_frames,
        num_detected_objects_threshold,
        obj_detection_confidence_threshold)

    return detections


def write_clip(clip_info: dict,
               save_path: str,
               file_name: str):
    save_path = f"{save_path}{file_name}"
    with open(save_path, 'wb') as clip_file:
        pickle.dump(clip_info, clip_file)
