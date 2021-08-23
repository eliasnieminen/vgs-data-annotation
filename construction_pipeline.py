import pickle

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

import od_utils
from env import ProjectEnvironment
from pathlib import Path
import construction_utils as cu
from video_metadata import VideoMetadata
from speech_noise_analyzer import YamNetSpeechNoiseAnalyzer
from util import format_id


# Initialize environment variable
env = ProjectEnvironment()

# Initialize speech noise analyzer
speech_noise_analyzer = YamNetSpeechNoiseAnalyzer()

# Only the videos with these suffixes are handled.
allowed_video_suffixes = [".mp4", ".mkv", ".webm"]

# Determine which dataset and split (train, test, val)
# the construction is performed on.
target_dataset = "crosstask"
target_split = "train"

# Determine the object detection parameters (if used)
num_detected_objects_threshold = 1
obj_detection_confidence_threshold = 0.3
model_name = "ssd_mobilenet_v1_coco_2017_11_17"

# If False, no object detection algorithm is applied.
use_object_detection = False

# If False, no audio content analysis is performed (it might be done before).
analyze_audio_content = False

# Determine whether the clips are selected by random or they are loaded from
# ready annotations.
use_random_clips = False

# If True, the best frames will be shown with matplotlib.pyplot.
plotting = False

# If 'path', the frames from the best frame detection
# are saved to a specified directory and only the image paths are returned.
best_frame_return_type = "path"

# ======================

# Get paths to every video in the dataset split.
video_paths = cu.get_dataset_video_paths(
    target_dataset=target_dataset,
    target_split=target_split,
    allowed_video_suffixes=allowed_video_suffixes)

# This will contain all object detections if the object detection is enabled.
all_detections = {}

# This will contain all the clips and the best frames from those clips.
all_clips_and_frames = {}

# DEV: For testing purposes, one can limit the max number of videos to be
# processed. If None, there will be no limit.
max_videos_limit = 2

# For keeping track of the video count.
count = 0

# The main loop
for video_path in video_paths:
    # Early stopping.
    if max_videos_limit is not None and count == max_videos_limit:
        break
    print(f"Processing {video_path}...")

    # Determine video variables.
    vid_metadata = cu.get_video_metadata(video_path,
                                         original_dataset=target_dataset)
    dur = vid_metadata.duration
    fps = vid_metadata.fps
    frame_count = vid_metadata.frame_count
    metadata = vid_metadata.metadata

    # Determine the save path.
    frame_save_path = f"{env['clip_save_path']}" \
                      f"{target_dataset}/{target_split}/frames/"

    # Get the clips.
    # If random clips, one can determine properties of the clips.
    # If not random, the clip data will be loaded from file.
    if use_random_clips:
        # The first list contains the timestamps in seconds, the second list
        # contains the same timestamps as video frame numbers.
        clip_list, clip_list_as_frames = cu.get_random_clips_as_list(
            n_clips=2,
            clip_length=2.0,
            video_metadata=vid_metadata,
            sort_clips=True)
        clip_infos = None
    else:
        # The first list contains the timestamps in seconds, the second list
        # contains the same timestamps as video frame numbers. The third list
        # contains information of the clips.
        try:
            clip_list, clip_list_as_frames, clip_infos \
                = cu.get_annotated_clips_as_list(
                    yt_id=metadata["yt_id"],
                    target_dataset=target_dataset,
                    target_split=target_split)
        except ValueError:
            print("No annotations found, skipping file.")
            continue

    # Will contain the best frames for each clip either as a numpy array or
    # file path (recommended).
    clips_best_frames = {}

    # DEV: For testing purposes, one can limit the amount of clips processed
    # per video.
    clip_lim = None
    clip_count = 0

    # Go through the clips.
    for i in range(len(clip_list)):
        # Early stopping.
        if clip_lim is not None and clip_count == clip_lim:
            break

        print(f"Processing clip {i + 1} / {len(clip_list)}...")

        # Get the start and end of the clip as seconds.
        start_t, end_t = clip_list[i]

        # Get the start and end of the clip as frames.
        start_f, end_f = clip_list_as_frames[i]

        # Select best frames from the current clip (1 per second).
        frames = cu.select_sharpest_frames(video_path,
                                           start_f,
                                           end_f,
                                           vid_metadata,
                                           plotting=plotting,
                                           return_type=best_frame_return_type,
                                           clip_id=i,
                                           save_dir=frame_save_path)

        # Process the audio contents (determine speech / noise proportion)
        if analyze_audio_content:
            sn_ratio = speech_noise_analyzer.analyze(video_path,
                                                     start_t=start_t,
                                                     end_t=end_t)
        # If no proportion is calculated.
        else:
            sn_ratio = None

        # If the speech noise proportion was already calculated, save it.
        if clip_infos is not None:
            clip_info = clip_infos[i]
            sn_ratio \
                = clip_info["segment"]["speech_noise_ratio"]
            description = clip_info["step_description"]
            step_num = clip_info["step_num"]
            clip_num = clip_info["clip_num"]
            task_id = clip_info["task_id"]
        else:
            description = None
            step_num = None
            clip_num = i
            task_id = None

        all_info = {
            "frames": frames,
            "clip_metadata": {
                "clip_num": clip_num,
                "task_id": task_id,

                "original_file": video_path,
                "original_dataset": target_dataset,

                "frame_type": best_frame_return_type,

                "segment": {
                    "step_num": step_num,
                    "step_description": description,

                    "segment_t": (start_t, end_t),
                    "segment_f": (start_f, end_f),

                    "detections": 0,
                    "speech_noise_ratio": sn_ratio,
                },
            }
        }

        clips_best_frames[i] = all_info

        clip_count += 1

    all_clips_and_frames[vid_metadata.metadata["yt_id"]] = clips_best_frames

    count += 1

if use_object_detection:

    print("Loading the model... This might take a while.")
    # Load model. Takes up to 15 seconds, in debug up to 35 seconds.
    model_loading_start = time.time()
    obj_detection_model = od_utils.load_model(model_name)
    model_loading_end = time.time()
    print(f"Model loading took "
          f"{'{:.2f}'.format(model_loading_end - model_loading_start)} seconds.")

    # Get object detecions for each video's clip's frames.
    detections = cu.detect_objects(all_clips_and_frames,
                                   obj_detection_model,
                                   num_detected_objects_threshold,
                                   obj_detection_confidence_threshold)

    # Assign detections to each of the clips.
    for yt_id_det in detections.keys():
        for yt_id_clip in all_clips_and_frames.keys():
            if yt_id_det == yt_id_clip:
                for clip in all_clips_and_frames[yt_id_det].keys():
                    all_clips_and_frames[yt_id_clip][clip]["detections"] \
                        = detections[yt_id_det][clip]

# Save the obtained clips and frames.
count = 0
for yt_id in all_clips_and_frames.keys():
    clips = all_clips_and_frames[yt_id]
    for clip_num in clips.keys():
        clip = clips[clip_num]
        save_dir = f"{env['clip_save_path']}{target_dataset}/{target_split}/"
        clip_id = format_id(count)
        file_name = f"{clip_id}_{yt_id}_clip{clip_num}.pickle"
        save_path = f"{save_dir}{file_name}"
        with open(save_path, 'wb') as clip_file:
            pickle.dump(clip, clip_file)
        count += 1
