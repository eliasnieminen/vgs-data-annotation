import numpy as np
import src.utilities.construction_utils as cu
import shutil
import warnings
import matplotlib

from pathlib import Path
from src.utilities.env import ProjectEnvironment
from numpy.random import default_rng
from src.analysis.speech_noise_analyzer import YamNetSpeechNoiseAnalyzer
from src.utilities.clipper import Clipper
from audioread import NoBackendError

matplotlib.use("TkAgg")

env = ProjectEnvironment()

# Select 100 videos (randomly)

# Do clipping based on annotations
# Do clipping based on Yamnet analysis

allowed_video_suffixes = [".mp4", ".mkv", ".webm"]

target_dataset = "youcook2"
target_split = "train"
override_video_path = False

num_random_videos = 8

random_clips_per_minute = 3
random_clips_per_second = random_clips_per_minute / 60
random_clip_lenght = 10.0

num_trials = 40
rand_nums = np.random.rand(num_trials)

experiment = True

sn_analyzer = YamNetSpeechNoiseAnalyzer()
speech_noise_thresholds = [1.0, 0.9, 0.8]
snr_threshold = 0.8

if not override_video_path:

    try:
        video_path = env[f"{target_dataset}_save_path"] \
            if target_split == "train" \
            else env[f"{target_dataset}_save_path_{target_split}"]

    except KeyError:
        raise KeyError(f"Invalid split '{target_split}' "
                       f"for this dataset: '{target_dataset}'")

else:
    video_path = "/lustre/scratch/specog/youcook2_dataset/train/"

# clips_save_path = f"/lustre/scratch/specog/youcook2_analysis/" \
                  # f"{target_dataset}/{target_split}/"

clips_save_path = f"subjective_analysis/clip_selection_comparison/" \
                  f"{target_dataset}/{target_split}/"

# Get all available video paths.
all_video_paths = []
file_count = 0
for file in Path(video_path).resolve().iterdir():
    if file.suffix in allowed_video_suffixes:
        all_video_paths.append(str(file))
        file_count += 1

# Get random video paths from all video paths.
num_videos = len(all_video_paths)
rng = default_rng()
random_video_indexes = rng.choice(num_videos, size=file_count, replace=False)

all_video_paths = np.array(all_video_paths)
video_paths = all_video_paths[random_video_indexes]
video_paths: list = np.ndarray.tolist(video_paths)

video_lim = None
video_count = 0

bad_file = False

for file in video_paths:
    if video_lim is not None and "228" not in file:
        continue

    if video_lim is not None and video_count == video_lim:
        break

    print(str(file))

    bad_file = False
    final_clips = []

    video_metadata = cu.get_video_metadata(file, target_dataset)
    duration = video_metadata.duration
    yt_id = video_metadata.metadata["yt_id"]

    if Path(f"{clips_save_path}{yt_id}").exists():
        print("Video has already been processed. Skipping.")
        continue

    seconds: np.ndarray = np.arange(0, int(duration))
    np.random.shuffle(seconds)
    seconds: list = np.ndarray.tolist(seconds)

    clipper = Clipper()
    clipper.load(file)

    # num_clips = np.round(random_clips_per_second * duration).astype(
    #     np.int16)
    num_clips_whole = np.round(random_clips_per_second * duration).astype(
        np.int16)

    num_clips = 1
    num_accepted_clips = 0
    accepted_clip_nums = []

    for second in seconds:

        print(f"Second {second}")

        rand_clip = (float(second), float(second) + random_clip_lenght)

        print(f"Current number of accepted clips: "
              f"{num_accepted_clips} / {num_clips_whole}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                speech_proportion, noise_proportion = sn_analyzer.analyze(
                    file,
                    start_t=rand_clip[0],
                    end_t=rand_clip[1])
        except NoBackendError:
            print(f"Bad file: {file}. Skipping.")
            bad_file = True
            break

        if speech_proportion >= snr_threshold:

            file_name = f"{yt_id}_randomclip{num_accepted_clips}_" \
                        f"SNR{int(speech_proportion * 100)}_sec{second}"
            save_path = f"{clips_save_path}{yt_id}/yamnet/"

            # Create save dir if it doesn't exist.
            Path(save_path).mkdir(parents=True, exist_ok=True)

            clipper.create_clip(save_path=save_path,
                                file_name=file_name,
                                start_t=rand_clip[0],
                                end_t=rand_clip[1])

            final_clips.append((rand_clip[0], rand_clip[1]))

            found = True
            num_accepted_clips += 1

            if num_accepted_clips == num_clips_whole:
                break

    if bad_file:
        continue

    if num_accepted_clips == num_clips_whole:
        print("Found all!")
    else:
        print("Missing clips.")

    try:
        ann_clips, ann_clips_as_frames, clip_infos \
            = cu.get_annotated_clips_as_list(yt_id=yt_id,
                                             target_dataset=target_dataset,
                                             target_split=target_split)
    except ValueError:
        print("Annotations not found for this video, skipping...")
        continue

    clip_count = 0
    clip_lim = None
    for i in range(len(ann_clips)):

        if clip_lim is not None and clip_count == clip_lim:
            break

        ann_clip = ann_clips[i]

        start_t = ann_clip[0]
        end_t = ann_clip[1]

        save_path = f"{clips_save_path}{yt_id}/annotation/"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        file_name = f"{yt_id}_annotatedclip{clip_count}"
        txt_file_name = f"{yt_id}_annotatedclip{clip_count}_annotation.txt"
        clipper.create_clip(save_path=save_path,
                            file_name=file_name,
                            start_t=start_t,
                            end_t=end_t)
        with open(f"{save_path}{txt_file_name}", 'w') as txt_file:
            txt_file.write(clip_infos[i]["step_description"])

        clip_count += 1

    shutil.copy(file, f"{clips_save_path}{yt_id}/{yt_id}.mp4")
