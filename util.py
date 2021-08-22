import time

import numpy as np
from numpy.random import default_rng
from typing import Optional, Union
from pathlib import Path
import os
import signal
import subprocess
import cv2 as cv


def get_random_rows_from_file(file_path: str,
                              n_rows: Optional[int] = 100):
    """This function returns random rows from a specified file.

    Args:
      file_path: The specified file
      n_rows: The amount of rows to be returned. If the number exceeds the
    number of rows in the specified file, a ValueError will be
    thrown.

    Returns:
      List of rows.

    """

    try:
        rows = []
        with open(file_path, 'r') as file:
            rows = file.readlines()
        # If the desired num of rows exceeds the number of available rows.
        if len(rows) < n_rows:
            raise ValueError
        else:
            num_rows = len(rows)
            rng = default_rng()
            random_indexes = rng.choice(np.arange(num_rows),
                                        size=n_rows,
                                        replace=False)
            rows = np.array(rows)
            ret = rows[random_indexes]
            ret = np.ndarray.tolist(ret)
            return ret
    except ValueError:
        raise ValueError("n_rows exceeds the amount of "
                         "rows in the specified file.")
    except FileNotFoundError:
        raise FileNotFoundError("The specified file was not found.")


def check_file_exists(file_name):
    """Checks if a video file exists.

    Args:
      file_name: The searched file.

    Returns:
      True, if exists. False, if doesn't exist.

    """
    if os.path.exists(file_name) or \
       os.path.exists(file_name.replace(".mp4", ".mkv")) or \
       os.path.exists(file_name.replace(".mp4", ".mp4.mkv")) or \
       os.path.exists(file_name.replace(".mp4", ".mp4.webm")):
        return True
    else:
        return False


def download_video(
        save_path: str,
        video_id: Union[int, str],
        video_yt_id: str,
        yt_dl_executable: Optional[str] = "c:/tools/youtube-dl/"
                                          "youtube-dl.exe",
        max_waiting_time: Optional[int] = 200) \
        -> Union[bool, None]:
    """Downloads a video from YouTube with the specified YouTube video id. The
    function uses a subprocess that start the 'youtube-dl' program outside
    of the current running program. The download stops if the maximum waiting
    time is exceeded.

    Args:
      save_path: The location to which the downloaded video is saved.
      video_id: The numerical id of the video (e.g. a task number)
      video_yt_id: The YouTube id of the video.
      yt_dl_executable: The location of the 'youtube-dl' executable. This is sometimes needed.
      max_waiting_time: The maximum waiting time for one video.
      Sometimes the loading gets stuck or is very slow, in which
      case the download is skipped to save time.

    Returns:
      None, if the download is skipped.
      True, if the download was successful.
      False, if the download failed.

    """
    save_name = f"{save_path}{video_id}_{video_yt_id}.mp4"
    # Check if the video is already downloaded earlier.
    if check_file_exists(save_name):
        print(f"File already exists: {save_name}, skipping...")
        return None

    yt_link = f"https://youtube.com/watch?v={video_yt_id}"

    use_shell = True if os.name == "posix" else False
    # Open the downloader subprocess.
    dl_proc = subprocess.Popen(f"{yt_dl_executable} "
                               f"-o \"{save_name}\" "
                               f"{yt_link}",
                               shell=use_shell)

    # Sometimes youtube-dl takes up to hours to download one video.
    # To prevent the massive amount of time wasted on single videos,
    # the process running time is monitored and the process is killed,
    # if it takes too long time to run.
    waiting_time = 0
    while dl_proc.poll() is None:
        time.sleep(1)
        waiting_time += 1  # Add a second to the waiting time.
        if waiting_time > max_waiting_time:
            print(f"Waited for {waiting_time} seconds and exceeded "
                  f"maximum waiting time. The process is now terminated.")
            dl_proc.terminate()
            # os.killpg(os.getpgid(dl_proc.pid), signal.SIGTERM)
            return None

    # If the download failed, the check_file_exists will return False.
    # Otherwise, it will return True.
    return check_file_exists(save_name)


def format_2f(f: float) -> str:
    """Formats a float to 2 decimal places.

    Args:
      f: The float to be formatted.

    Returns:
      The formatted float as string.

    """
    return '{:.2f}'.format(f)


def format_id(i: int, e: Optional[int] = 8) -> str:
    """Formats an integer with zero padding of desired number. (e.g. 1 => 00001)

    Args:
      i: The integer to be formatted
      e: The length of the final string.

    Returns:
      The formatted integer.

    """
    num_str = str(i)
    while len(num_str) < e:
        num_str = "0" + num_str

    return num_str


def save_frame(save_dir: str,
               frame: np.ndarray,
               yt_id: str,
               clip_id: int,
               frame_id: int) -> str:
    """Saves a frame as an image to the specified directory.

    Args:
      save_dir: The directory where the frame should be saved.
      frame: The frame to be saved.
      yt_id: The YouTube id of the video that the frame is from.
      clip_id: The id of the random clip that the frame is from.
      frame_id: The id of the frame.

    Returns:
      The ultimate file path that was saved.

    """
    file_name = f"{yt_id}_clip{clip_id}_frame{frame_id}.jpg"
    save_path = save_dir + file_name
    cv.imwrite(save_path, frame)
    return save_path


def get_row_by_yt_id(csv_path: str, yt_id: str):
    """

    Args:
        csv_path:
        yt_id:

    Returns:

    """
    csv_file = open(csv_path, 'r')
    rows = [line.strip() for line in csv_file.readlines()]
    for row in rows:
        if yt_id in row:
            return row

    return None
