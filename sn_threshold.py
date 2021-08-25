import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle

import construction_utils as cu
from speech_noise_analyzer import YamNetSpeechNoiseAnalyzer
from env import ProjectEnvironment
from pathlib import Path
from util import format_2f

# matplotlib.use("TkAgg")

env = ProjectEnvironment()
sn_analyzer = YamNetSpeechNoiseAnalyzer()

target_dataset = "youcook2"
target_split = "train"

allowed_video_suffixes = [".mp4", ".mkv", ".webm"]

video_path = None
override_video_path = True

random_clips_per_minute = 6
random_clips_per_second = random_clips_per_minute / 60
random_clip_length = 10.0

if not override_video_path:

    try:
        video_path = env[f"{target_dataset}_save_path"] \
            if target_split == "train" \
            else env[f"{target_dataset}_save_path_{target_split}"]

    except KeyError:
        print(f"Invalid split '{target_split}' "
              f"for this dataset: '{target_dataset}'")

else:
    video_path = "/lustre/scratch/specog/youcook2_dataset/train/"


file_count = 0
for file in Path(video_path).resolve().iterdir():
    if file.suffix in allowed_video_suffixes:
        file_count += 1

print(f"Found {file_count} videos from {video_path}.")

speech_proportions = []
speech_proportions_per_file = {}

check_previous = True
session_id = format_2f(time.time())

if check_previous:

    previously_calculated_path = "misc/dist/"
    prev_id = "1629798870.00"

    for file in Path(previously_calculated_path).resolve().iterdir():
        if prev_id in file.name:
            with open(str(file), 'rb') as f:
                if "per" in file.name:
                    speech_proportions_per_file = pickle.load(f)
                else:
                    speech_proportions = pickle.load(f)

    if len(speech_proportions) > 0 and len(speech_proportions_per_file) > 0:
        session_id = prev_id
        print("Found previous computations. Continuing.")
    else:
        print("No previous computations found. Starting from beginning.")

lim = None
count = 0

for file in Path(video_path).iterdir():
    timing_start = time.time()
    
    if lim is not None and count == lim:
        break

    if file.suffix in allowed_video_suffixes:
        
        if ".f135" in file.name or ".f133" in file.name:
            print("Faulty file, skipping...")
            count += 1
            continue
        
        print(f"Processing file {count + 1} / {file_count}.")
        
        video_metadata = cu.get_video_metadata(str(file), target_dataset)
        yt_id = video_metadata.metadata["yt_id"]

        if yt_id in speech_proportions_per_file.keys():
            print("Already calculated, skipping...")
            count += 1
            continue

        n_clips = np.round(video_metadata.duration * random_clips_per_second).astype(np.int16)

        clips, clips_as_frames = cu.get_random_clips_as_list(
            n_clips=n_clips,
            clip_length=random_clip_length,
            video_metadata=video_metadata
        )

        speech_proportions_per_file[yt_id] = []

        clip_count = 0
        for clip in clips:
            print(f"Clip {clip_count + 1} / {len(clips)}")
            sn_ratio \
                = sn_analyzer.analyze(video=str(file),
                                      start_t=clip[0],
                                      end_t=clip[1])

            if sn_ratio is not None:
                speech_proportion, noise_proportion = sn_ratio
            else:
                print("Error reading file, skipping...")
                continue

            speech_proportions.append(speech_proportion)
            speech_proportions_per_file[yt_id].append(speech_proportion)
            clip_count += 1

        timing_end = time.time()

        print(f"{str(file)} file took {format_2f(timing_end - timing_start)} seconds.")

        with open(f"misc/dist/speech_proportions_{target_dataset}_{target_split}_{session_id}.pickle",
                  'wb') as f:
            pickle.dump(speech_proportions, f)

        with open(f"misc/dist/speech_proportions_per_file_{target_dataset}_{target_split}_{session_id}.pickle",
                  'wb') as f:
            pickle.dump(speech_proportions_per_file, f)

        count += 1

print(f"Processed {count} videos.")


speech_proportions = np.array(speech_proportions)

bins = np.linspace(0, 1, num=50)

plt.hist(speech_proportions, bins)

plt.savefig(f"figures/sn_ratio_distribution/dist1_{session_id}.png")
plt.show()
