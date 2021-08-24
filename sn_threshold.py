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

matplotlib.use("TkAgg")

env = ProjectEnvironment()
sn_analyzer = YamNetSpeechNoiseAnalyzer()

target_dataset = "youcook2"
target_split = "train"

allowed_video_suffixes = [".mp4", ".mkv", ".webm"]

video_path = None

random_clips_per_minute = 6
random_clips_per_second = random_clips_per_minute / 60
random_clip_length = 10.0

try:
    video_path = env[f"{target_dataset}_save_path"] \
        if target_split == "train" \
        else env[f"{target_dataset}_save_path_{target_split}"]

except KeyError:
    print(f"Invalid split '{target_split}' "
          f"for this dataset: '{target_dataset}'")

speech_proportions = []

lim = None
count = 0

for file in Path(video_path).iterdir():
    timing_start = time.time()

    if lim is not None and count == lim:
        break

    if file.suffix in allowed_video_suffixes:

        video_metadata = cu.get_video_metadata(str(file), target_dataset)

        n_clips = np.round(video_metadata.duration * random_clips_per_second).astype(np.int16)

        clips, clips_as_frames = cu.get_random_clips_as_list(
            n_clips=n_clips,
            clip_length=random_clip_length,
            video_metadata=video_metadata
        )

        clip_count = 0
        for clip in clips:
            print(f"Clip {clip_count + 1} / {len(clips)}")
            speech_proportion, noise_proportion \
                = sn_analyzer.analyze(video=str(file),
                                      start_t=clip[0],
                                      end_t=clip[1])

            speech_proportions.append(speech_proportion)
            clip_count += 1

    timing_end = time.time()

    print(f"One file took {format_2f(timing_end - timing_start)} seconds.")

    count += 1

print(f"Processed {count} videos.")

time_id = format_2f(time.time())

with open(f"misc/dist/speech_proportions_{time_id}.pickle",
          'wb') as f:
    pickle.dump(speech_proportions, f)

speech_proportions = np.array(speech_proportions)

bins = np.linspace(0, 1, num=50)

plt.hist(speech_proportions, bins)

plt.savefig(f"figures/sn_ratio_distribution/dist1_{time_id}.png")
plt.show()
