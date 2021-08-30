import numpy as np
import construction_utils as cu
import annotation_utils as au
import shutil
import warnings
import matplotlib
import matplotlib.pyplot as plt
import librosa as lbr

from pathlib import Path
from env import ProjectEnvironment
from numpy.random import default_rng
from speech_noise_analyzer import YamNetSpeechNoiseAnalyzer
from clipper import Clipper
from audioread import NoBackendError
from util import format_2f

matplotlib.use("TkAgg")

env = ProjectEnvironment()

# Select 100 videos (randomly)

# Do clipping based on annotations
# Do clipping based on Yamnet analysis

for a in range(1):
    print(f"Run {a + 1} / 3")

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

    experiment = False

    sn_analyzer = YamNetSpeechNoiseAnalyzer()
    speech_noise_thresholds = [1.0, 0.9, 0.8]

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

    clips_save_path = "/lustre/scratch/specog/youcook2_analysis/"

    # Get all available video paths.
    all_video_paths = []
    for file in Path(video_path).resolve().iterdir():
        if file.suffix in allowed_video_suffixes:
            all_video_paths.append(str(file))

    # Get random video paths from all video paths.
    num_videos = len(all_video_paths)
    rng = default_rng()
    random_video_indexes = rng.choice(num_videos, size=num_random_videos, replace=False)

    all_video_paths = np.array(all_video_paths)
    video_paths = all_video_paths[random_video_indexes]
    video_paths: list = np.ndarray.tolist(video_paths)

    video_lim = 1
    video_count = 0

    bad_file = False

    for file in video_paths:
        if "228" not in file:
            continue

        if video_lim is not None and video_count == video_lim:
            break

        print(str(file))

        bad_file = False
        clip_onsets = []
        clip_orders = []

        video_metadata = cu.get_video_metadata(file, target_dataset)
        duration = video_metadata.duration
        yt_id = video_metadata.metadata["yt_id"]

        clipper = Clipper()
        clipper.load(file)

        # num_clips = np.round(random_clips_per_second * duration).astype(
        #     np.int16)
        num_clips_whole = np.round(random_clips_per_second * duration).astype(
            np.int16)

        num_clips = 1
        num_accepted_clips = 0
        accepted_clip_nums = []

        for trial_num in range(num_trials):

            print(f"Trial {trial_num + 1} / {num_trials}")

            found = False

            if not experiment:

                rand_clips, rand_clips_as_frames = cu.get_random_clips_as_list(
                    n_clips=num_clips,
                    clip_length=random_clip_lenght,
                    video_metadata=video_metadata)

                rand_clip = rand_clips[0]

            else:
                start = rand_nums[trial_num] * video_metadata.duration - random_clip_lenght
                rand_clip = (start, start + random_clip_lenght)

            for speech_noise_threshold in speech_noise_thresholds:

                print(f"Processing SNR threshold of {speech_noise_threshold}")

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

                if speech_proportion >= speech_noise_threshold:  # Accept
                    # file_name = f"{yt_id}_randomclip{num_accepted_clips}_" \
                    #             f"SNR{int(speech_noise_threshold * 100)}"
                    # save_path = f"{clips_save_path}{yt_id}/yamnet/"
                    #
                    # # Create save dir if it doesn't exist.
                    # Path(save_path).mkdir(parents=True, exist_ok=True)
                    #
                    # clipper.create_clip(save_path=save_path,
                    #                     file_name=file_name,
                    #                     start_t=rand_clip[0],
                    #                     end_t=rand_clip[1])

                    clip_onsets.append(rand_clip[0])
                    clip_orders.append(num_accepted_clips)

                    found = True
                    num_accepted_clips += 1

                    if found:
                        break
                else:
                    print("Not accepted, next trial.")


            if bad_file:
                break

            if num_accepted_clips == num_clips_whole:
                break

        if bad_file:
            continue

        if num_accepted_clips == num_clips_whole:
            print("Found all!")
        else:
            print("Missing clips.")

        break

    sn_analyzer.analyze(video=file, whole_video=True)
    segments = sn_analyzer.get_segments()

    print(clip_onsets)
    print(clip_orders)

    audio, sr = lbr.load(file, sr=None)
    dur = lbr.get_duration(audio, sr)
    n = np.linspace(0, dur, num=len(audio))

    plt.figure(figsize=(10, 10))
    plt.plot(n, audio)

    for i in range(len(clip_orders)):
        clip_x = clip_onsets[i]
        clip_order = clip_orders[i]
        plt.vlines(clip_x, ymin=-1.5, ymax=1.5+((i % 10) * 0.07), color="r")
        plt.text(clip_x+0.05, 1.5+((i % 10) * 0.07), f"{clip_order}")

    plt.title(f"Khazar {a + 1} on {file}")

    plt.savefig(f"figures/rand/228_khazar{a}.png")
    plt.clf()

    #
    # try:
    #     ann_clips, ann_clips_as_frames, clip_infos \
    #         = cu.get_annotated_clips_as_list(yt_id=yt_id,
    #                                          target_dataset=target_dataset,
    #                                          target_split=target_split)
    # except ValueError:
    #     print("Annotations not found for this video, skipping...")
    #     continue
    #
    # clip_count = 0
    # clip_lim = None
    # for i in range(len(ann_clips)):
    #
    #     if clip_lim is not None and clip_count == clip_lim:
    #         break
    #
    #     ann_clip = ann_clips[i]
    #
    #     start_t = ann_clip[0]
    #     end_t = ann_clip[1]
    #
    #     save_path = f"{clips_save_path}{yt_id}/annotation/"
    #     Path(save_path).mkdir(parents=True, exist_ok=True)
    #
    #     file_name = f"{yt_id}_annotatedclip{clip_count}"
    #     txt_file_name = f"{yt_id}_annotatedclip{clip_count}_annotation.txt"
    #     clipper.create_clip(save_path=save_path,
    #                         file_name=file_name,
    #                         start_t=start_t,
    #                         end_t=end_t)
    #     with open(f"{save_path}{txt_file_name}", 'w') as txt_file:
    #         txt_file.write(clip_infos[i]["step_description"])
    #
    #     clip_count += 1
    #
    # shutil.copy(file, f"{clips_save_path}{yt_id}/{yt_id}.mp4")
