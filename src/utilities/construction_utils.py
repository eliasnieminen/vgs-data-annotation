import warnings

import cv2 as cv
import math

import numpy as np
import src.utilities.od_utils
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.utilities import util
import pickle

from src.utilities.env import ProjectEnvironment
from pathlib import Path
from typing import Optional, List, Union
from src.utilities.video_metadata import VideoMetadata
from src.analysis.image_quality import BlurDetector
from src.utilities.clipper import Clipper
from audioread import NoBackendError
from src.analysis.speech_noise_analyzer import YamNetSpeechNoiseAnalyzer

env = ProjectEnvironment()


def get_dataset_video_paths(target_dataset: str,
                            target_split: str,
                            allowed_video_suffixes: List):
    """Returns the paths to all videos in the given target dataset.

    Args:
        target_dataset: The target dataset.
        target_split: The target split of the dataset.
        allowed_video_suffixes: The allowed video file extensions that will
                                be included.

    Returns: The list of dataset split videos.

    """

    dataset_save_path = env["base_path"] + env[f"{target_dataset}_save_path"] \
        if target_split == "train" \
        else env[f"{target_dataset}_save_path_{target_split}"]

    video_paths = []
    for file in Path(dataset_save_path).resolve().iterdir():
        if file.suffix in allowed_video_suffixes:
            video_paths.append(str(file))

    return video_paths


def get_yt_id_from_video_path(video_path):
    """Gets the YouTube id from a file path.

    Args:
        video_path: The video path.

    Returns: The YouTube id as string.

    """
    p = Path(video_path).resolve()
    stem = p.stem.replace(".mp4", "")
    spl = stem.split("_")
    yt_id = "_".join(spl[1:])
    return yt_id


def get_task_id_from_video_path(video_path: str):
    """Gets the task id from a video path.

    Args:
        video_path: The video path. (assumed to contain task id.)

    Returns: The task id

    """
    p = Path(video_path).resolve()
    stem = p.stem.replace(".mp4", "")
    spl = stem.split("_")
    task_id = spl[0]
    return task_id


def get_video_metadata(video_path: str, original_dataset: str):
    """Returns metadata for the given video path using OpenCV.

    Args:
        video_path: The path to the video.
        original_dataset: The dataset of the video.

    Returns: A VideoMetadata object containing various data about the video.

    """
    # Get some video data.
    yt_id = get_yt_id_from_video_path(video_path)
    task_id = get_task_id_from_video_path(video_path)

    # Open the OpenCV video capture.
    cap = cv.VideoCapture(video_path)

    # Wait for the file.
    while not cap.isOpened():
        cap = cv.VideoCapture(video_path)
        cv.waitKey(1000)
        print("Waiting for the header...")

    # Get fps and duration from OpenCV
    fps = float("{:.2f}".format(cap.get(cv.CAP_PROP_FPS)))
    dur = math.floor(cap.get(cv.CAP_PROP_FRAME_COUNT) / fps)

    # Construct the object.
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
    """Generates a fully random list of timestamps based on the input parameters.

    Args:
        n_clips: How many clips?
        clip_length: How long clips?
        video_metadata: The metadata of the video from which the timestamps
                        are generated.
        sort_clips: If True, the clips will be sorted by the starting onset.

    Returns: Tuple: (List of clips: (start, end) as seconds,
                     List of clips: (start, end) as frame numbers)

    """
    fps = video_metadata.fps

    # Generate the offsets.
    starts = np.random.rand(n_clips) * (video_metadata.duration - clip_length)
    # Convert the timestamps to frames.
    starts_as_frames = np.round(starts * fps).astype(np.int64)

    # Sort clips if so desired.
    if sort_clips:
        starts = np.sort(starts)
        starts_as_frames = np.sort(starts_as_frames)

    clips = []
    clips_as_frames = []

    # Create the clip tuplets for time and frame format respectively.
    for s in starts:
        clips.append((s, s + clip_length))
    for f in starts_as_frames:
        clips_as_frames.append(
            (f, f + np.round(clip_length * fps).astype(np.int64)))

    return clips, clips_as_frames


def get_random_clips_as_list_v2(video_metadata: VideoMetadata,
                                clips_per_minute: Optional[int] = 3,
                                clip_length: Optional[float] = 10.0,
                                snr_threshold: Optional[float] = 0.8):
    """Generates random clips and analyzes their audio contents, returns a
    list of random clips that have SNR over the snr_threshold.

    Args:
        video_metadata: The metadata of the video that will be processed.
        clips_per_minute: The amount of random clips generated per minute on
                          average.
        clip_length: The random clip length.
        snr_threshold: The SNR threshold for selecting a clip. The SNR will be
                       computed for all of the clips and the ones that have
                       a higher or equal threshold will be accepted to be
                       included in the training data.

    Returns: Tuple: (List of clips: (start, end) as seconds,
                     List of clips: (start, end) as frame numbers)

    """
    file = video_metadata.metadata["filename"]
    final_clips = []
    final_clips_as_frames = []

    # For computing the amount of random clips that will be returned.
    random_clips_per_second = clips_per_minute / 60

    duration = video_metadata.duration

    # Creates a list of all possible onsets at the precision of one second.
    # The list will be shuffled and the random onsets will be used as the
    # starting points for random clips. The list will be iterated until
    # a sufficient amount of acceptable clips have been found.
    seconds: np.ndarray = np.arange(0, int(duration))
    np.random.shuffle(seconds)
    seconds: list = np.ndarray.tolist(seconds)

    # The complete number of clips that will be returned.
    num_clips_whole = np.round(
        random_clips_per_second * duration).astype(np.int16)

    # The current number of accepted clips.
    num_accepted_clips = 0

    # The analyzer class that will be used to analyze the audio contents.
    sn_analyzer = YamNetSpeechNoiseAnalyzer()

    # Go through the random onsets in the second-list.
    for second in seconds:

        print(f"Second {second}")

        rand_clip = (float(second), float(second) + clip_length)

        print(f"Current number of accepted clips: "
              f"{num_accepted_clips} / {num_clips_whole}")

        try:
            # PySound fails which will print out a warining.
            # The warnings are suppressed.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Compute the speech and noise proportions with the analyzer
                # class.
                speech_proportion, noise_proportion = sn_analyzer.analyze(
                    file,
                    start_t=rand_clip[0],
                    end_t=rand_clip[1])
        # Some files are corrupted and will cast an error, those files will
        # be skipped.
        except NoBackendError:
            print(f"Bad file: {file}. Skipping.")
            bad_file = True
            break

        # Check if the speech proportion of the clip is acceptable.
        if speech_proportion >= snr_threshold:

            final_clips.append((rand_clip[0], rand_clip[1]))
            final_clips_as_frames.append(
                (int(rand_clip[0] * video_metadata.fps),
                 int(rand_clip[1] * video_metadata.fps)))

            num_accepted_clips += 1

            # The sufficient amount of clips have been found.
            # Break.
            if num_accepted_clips == num_clips_whole:
                break

    if len(final_clips) == 0:
        return None

    return final_clips, final_clips_as_frames


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
    """This function is in charge of selecting the best frames from the given
    clip.

    For each second in the video, the algorithm selects the frame with the
    least blurriness, or, in turn, the maximum sharpness value. This is
    determined via OpenCV's Laplacian variation: when the variation is high,
    there is a lot of sharp edges in the given frame, when the variation is
    low, there is less sharp edges in the frame.

    Args:
        video_path: The path to the video to be analyzed.
        start_f: The frame from which the analysis begins.
        end_f: The frame to which the analysis ends.
        video_metadata: The metadata of the video.
        blur_threshold: (currently not used)
        plotting: Option to draw the best frames via matplotlib.pyplot
        return_type: If 'numpy': The return list will contain numpy arrays.
                     If 'path': The function will write the best frames to file
                     and return the paths to these files.
        clip_id: The number of the current clip to be processed.
        save_dir: The directory to which the best frames will be saved in case
                  the return_type is set to 'path'.

    Returns: List of numpy arrays or a list of paths to image files.

    """

    # The OpenCV capture interface.
    cap = cv.VideoCapture(video_path)

    # Wait for the file to open, if not immediate.
    while not cap.isOpened():
        cap = cv.VideoCapture(video_path)
        print("Waiting...")

    # Set the capture cursor to the starting frame.
    cap.set(cv.CAP_PROP_POS_FRAMES, start_f)

    # Get the current position of the cursor. (basically the same as start_f)
    pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)

    # Initialize the blurriness detector.
    blur_detector = BlurDetector(threshold=blur_threshold)

    # Temporary container for one second at a time.
    latest_second_frames = []

    # The container for the best frames.
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
                print(f"Sec {pos_frame} count: {count}, start_f: "
                      f"{start_f} end_f: {end_f} done")
                latest_second_frames = []
                frame_count += 1
        else:
            # Keep track of the waiting time, in case there is a stuck frame.
            print(f"wait_cumul: {wait_cumul}")
            if wait_cumul >= patience:
                print("Breaking due to stuck frame...")
                break
            # If patience is not up, move the cursor back a frame and see
            # if the frame can be loaded after 250ms waiting time.
            cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("Waiting for frame...")
            wait_cumul += 250
            cv.waitKey(250)

        # The end is reached.
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


def write_clip_pickle(clip_info: dict,
                      save_path: str,
                      file_name: str):
    save_path = f"{save_path}{file_name}"
    with open(save_path, 'wb') as clip_file:
        pickle.dump(clip_info, clip_file)
