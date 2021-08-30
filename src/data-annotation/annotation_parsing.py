import json
from src.utilities import annotation_utils as au

from src.utilities.construction_utils import get_video_metadata
from src.utilities.env import ProjectEnvironment
from pathlib import Path
from src.utilities.clipper import Clipper
from src.analysis.speech_noise_analyzer import YamNetSpeechNoiseAnalyzer

# Load the environment variable.
env = ProjectEnvironment()

# Determine the target dataset and split.
target_dataset = "youcook2"
target_split = "train"

# Determine clip save type.
clip_save_types = ["video", "pickle"]
write_clips_as = clip_save_types[1]

# Determine the save path for the annotated clips.
save_path = f"{env['clip_annotations_save_path']}" \
            f"{target_dataset}/{target_split}/"

override_video_path = False

if not override_video_path:

    # Get the dataset split video path:
    try:
        video_path \
            = env[f"{target_dataset}_save_path"] if target_split == "train" \
            else env[f"{target_dataset}_save_path_{target_split}"]

    except KeyError:
        # If the split or dataset was not found, raise an error.
        raise KeyError(f"Error: '{target_split}' is an invalid dataset "
                       f"split for '{target_dataset}'.")

else:
    video_path = "/lustre/scratch/specog/youcook2_dataset/train/"

# Define the allowed video suffixes.
allowed_video_suffixes = [".mp4", ".mkv", ".webm"]

# CrossTask dataset processing.
if target_dataset == "crosstask":

    # Define the path to raw annotations that will be processed.
    raw_annotations_path = env[f"crosstask_path"] + "annotations/"

    # Go through every video in the defined dataset split.
    for file in Path(video_path).resolve().iterdir():
        if file.suffix in allowed_video_suffixes:
            print(f"Processing {str(file)}...")

            # Get video variables.
            video_metadata = get_video_metadata(str(file), target_dataset)

            yt_id = video_metadata.metadata["yt_id"]
            task_id = video_metadata.metadata["task_id"]
            fps = video_metadata.fps

            # Get the annotations as a dict by YouTube id.
            all_annotations = au.find_annotation_by_yt_id(
                target_dataset=target_dataset,
                annotation_path=raw_annotations_path,
                yt_id=yt_id)

            if all_annotations is None:
                print(f"Annotations not found for video ID '{yt_id}'. "
                      f"Skipping file.")
                continue

            # Get the task text for the current task_id.
            # The task_text is a list of rows obtained from either
            # 'data/crosstask_release/tasks_primary.txt' or
            # 'data/crosstask_release/tasks_related.txt'.
            # It is very hacky, but this is the best way to obtain step
            # descriptions at the moment for this dataset.
            task_text = au.get_task_text(target_dataset=target_dataset,
                                         task_id=task_id)

            clip_num = 0

            # Initialize the Clipper object, that will create video clips
            # based on the annotations, if the 'write_clips_as' variable
            # is set to 'video'.
            clipper = Clipper()
            clipper.load(str(file))

            # Initialize the speech noise analyzer.
            sn_analyzer = YamNetSpeechNoiseAnalyzer()

            # Go through the annotations and save them to the specified
            # directory. 'annotation' is a tuple: (step_id, start, end).
            for annotation in all_annotations:
                # For some tasks, the task steps are fewer than they should,
                # and therefore there is no description for each step.
                # In this case, the step description is set to unknown.
                try:
                    # Here, the step name/description is obtained from the
                    # task_text list of rows. The steps are described on the
                    # last row.
                    step_desc = task_text[-1].split(",")[int(annotation[0]) - 1]
                except IndexError:
                    step_desc = "unknown_step"

                # Get the start and end of the clip as seconds.
                start_t = annotation[1]
                end_t = annotation[2]

                # Get the start and end of the clip as frames.
                start_f = int(fps * start_t)
                end_f = int(fps * end_t)

                # Calculate the speech noise ratio.
                speech_noise_ratio = sn_analyzer.analyze(video=str(file),
                                                         start_t=start_t,
                                                         end_t=end_t)

                file_name = f"{task_id}_{yt_id}_clip{clip_num}_" \
                            f"step{annotation[0]}"

                # Write the annotations:
                #
                # If 'write_clips_as' is set to 'video', each clip will be
                # saved as .mp4.
                #
                # If 'write_clips_as' is set to 'pickle', each clip will be
                # saved as a reference, which could be later used to further
                # process the clips. This way is recommended, since there
                # will be artifacts in the video when clipped like this.
                # Additionally, when clips are saved as a reference, they
                # don't require much space from the hard disk.
                if write_clips_as == "video":
                    clipper.create_clip(save_path,
                                        file_name,
                                        start_t=start_t,
                                        end_t=end_t)
                elif write_clips_as == "pickle":
                    clip_info = {
                        "task_id": task_id,
                        "yt_id": yt_id,
                        "original_dataset": target_dataset,
                        "clip_num": clip_num,
                        "step_num": annotation[0],
                        "step_description": step_desc,
                        "segment": {
                            "segment_t": (start_t, end_t),
                            "segment_f": (start_f, end_f),
                            "speech_noise_ratio": speech_noise_ratio[0]
                        }
                    }
                    success = au.create_reference_clip(save_path=save_path,
                                                       file_name=file_name,
                                                       clip_info=clip_info)
                    if not success:
                        print(f"{save_path}/{file_name}.pickle: "
                              f"File already exists.")

                else:
                    raise ValueError("Unrecognized save type"
                                     f" '{write_clips_as}' for clip.")
                clip_num += 1

# YouCookII dataset processing.
elif target_dataset == "youcook2":
    # Determine the path to the YouCookII annotations.
    # YouCookII offers annotations as a json file, unlike CrossTask.
    annotations_file_path = env["youcook2_path"] + \
                            "youcookii_annotations_trainval.json"

    # Load the json as a python dict.
    with open(annotations_file_path, 'r') as annotations_json_file:
        all_annotations = json.load(annotations_json_file)

    # Take the 'database' key, which holds all of the information.
    all_annotations = all_annotations["database"]

    # Go through each video in the specified dataset split.
    for file in Path(video_path).resolve().iterdir():
        if file.suffix in allowed_video_suffixes:

            # Get video variables.
            video_metadata = get_video_metadata(str(file), target_dataset)

            yt_id = video_metadata.metadata["yt_id"]
            task_id = video_metadata.metadata["task_id"]
            fps = video_metadata.fps

            already_computed = False
            for ann_file in Path(save_path).resolve().iterdir():
                if yt_id in ann_file.name:
                    already_computed = True
                    break

            if already_computed:
                print("Already computed, skipping...")
                continue

            # Get the annotations for the current video.
            try:
                annotations = all_annotations[yt_id]
            except KeyError:
                print("Invalid id, skipping")
                continue
            
            # Get the segments for the current video.
            segments = annotations["annotations"]

            # Initialize the Clipper, in case the clips are saved as video.
            clipper = Clipper()
            clipper.load(str(file))

            # Initialize the speech noise analyzer.
            sn_analyzer = YamNetSpeechNoiseAnalyzer()
            clip_num = 0
            for segment in segments:

                # Get the start and end as seconds.
                start_t = float(segment["segment"][0])
                end_t = float(segment["segment"][1])

                # Get the start and end as frames.
                start_f = int(fps * start_t)
                end_f = int(fps * end_t)

                # Get necessary data from the annotation.
                # Numerical id of the segment.
                seg_id = segment["id"]
                # The actual description of what is happening in the clip.
                sentence = segment["sentence"]

                # Calculate the speech noise ratio.
                speech_noise_ratio = sn_analyzer.analyze(video=str(file),
                                                         start_t=start_t,
                                                         end_t=end_t)

                file_name = f"{task_id}_{yt_id}_clip{clip_num}_" \
                            f"step{seg_id}"

                # See the comment in the CrossTask section for information
                # about the clip saving.
                if write_clips_as == "video":
                    clipper.create_clip(save_path=save_path,
                                        file_name=file_name,
                                        start_t=start_t,
                                        end_t=end_t)
                elif write_clips_as == "pickle":
                    clip_info = {
                        "task_id": task_id,
                        "yt_id": yt_id,
                        "original_dataset": target_dataset,
                        "clip_num": clip_num,
                        "step_num": seg_id,
                        "step_description": sentence,
                        "segment": {
                            "segment_t": (start_t, end_t),
                            "segment_f": (start_f, end_f),
                            "speech_noise_ratio": speech_noise_ratio[0]
                        }
                    }
                    success = au.create_reference_clip(save_path=save_path,
                                                       file_name=file_name,
                                                       clip_info=clip_info)
                    if not success:
                        print(f"{save_path}/{file_name}.pickle: "
                              f"File already exists.")
                else:
                    raise ValueError(f"Unrecognized save type "
                                     f"'{write_clips_as}' for clip.")

                clip_num += 1
