import pickle

from pathlib import Path
from env import ProjectEnvironment
from src.utilities.util import check_file_exists

env = ProjectEnvironment()


def find_annotation_by_yt_id(target_dataset: str,
                             annotation_path: str,
                             yt_id: str):
    """Finds annotation from file (for CrossTask dataset currently).

    Args:
        target_dataset: Crosstask.
        annotation_path: The path to crosstask annotation folder.
        yt_id: The YouTube id of the searched video.

    Returns: List of tuples: (step_num, step_start, step_end) as seconds.

    """

    annotation = ""

    if target_dataset == "crosstask":

        for file in Path(annotation_path).resolve().iterdir():
            if file.suffix == ".csv":
                if yt_id in file.name:
                    with open(str(file), 'r') as annotation_file:
                        annotation = annotation_file.read()
                    break

        if annotation == '':
            return None

        annotations = []
        rows = annotation.split("\n")
        for row in rows:
            if row == "":
                continue
            spl = row.split(",")
            step_num = int(spl[0])
            step_start = float(spl[1])
            step_end = float(spl[2])
            step = (step_num, step_start, step_end)
            annotations.append(step)

        return annotations


def get_task_text(target_dataset: str,
                  task_id: str):
    """For CrossTask. Gets the step description for the given task id.

    Args:
        target_dataset: Crosstask
        task_id: The id of the task.

    Returns: List of rows containing the task text description and info.

    """

    if target_dataset == "crosstask":
        primary_task_desc_file = env["crosstask_path"] + "tasks_primary.txt"
        related_task_desc_file = env["crosstask_path"] + "tasks_related.txt"

        found_flag = False
        count = 0
        task_txt = []

        primary_file = open(primary_task_desc_file)
        for row in primary_file.readlines():
            row = row.replace("\n", "")
            if not found_flag:
                if task_id in row:
                    found_flag = True
                    task_txt.append(row)
            else:
                if count < 4:
                    task_txt.append(row)
                    count += 1
                else:
                    break

        primary_file.close()

        if not found_flag:
            secondary_file = open(related_task_desc_file, 'r')
            for row in secondary_file.readlines():
                row = row.replace("\n", "")
                if not found_flag:
                    if task_id in row:
                        found_flag = True
                        task_txt.append(row)
                else:
                    if count < 4:
                        task_txt.append(row)
                        count += 1
                    else:
                        break
            secondary_file.close()

        return task_txt


def create_reference_clip(save_path: str,
                          file_name: str,
                          clip_info: dict):
    """Writes an annotation pickle file to desired location.

    Args:
        save_path: The destination directory.
        file_name: The file name of the pickle-file.
        clip_info: The object to be saved.

    Returns: None

    """
    file_name = file_name + ".pickle"

    final_save_path = f"{save_path}/{file_name}"
    if check_file_exists(final_save_path):
        return False

    with open(final_save_path, 'wb') as file:
        pickle.dump(clip_info, file)

    return True
