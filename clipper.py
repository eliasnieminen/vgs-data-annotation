import os
from typing import Optional, Union, Dict, List

import numpy as np
from pathlib import Path
import os
import subprocess
from env import ProjectEnvironment


class Clipper:
    """

    """
    def __init__(self, save_path: Optional[Union[str, None]] = None):
        self.clips: Dict[str, List] = {}
        self.env = ProjectEnvironment()
        self.current_file = None
        self.video_duration = None
        self.save_path = save_path

    def create_random_clips(self,
                            n_clips: int,
                            clip_length: float,
                            target_dataset: str,
                            verbose: Optional[int] = 1,
                            use_ffmpeg: Optional[bool] = True,
                            write_clips: Optional[bool] = True):
        """

        Args:
            n_clips:
            clip_length:
            target_dataset:
            verbose:
            use_ffmpeg:
            write_clips:

        Returns:

        """

        self.check_me()

        video_file_name = Path(self.current_file).resolve().stem
        spl = video_file_name.split("_")

        if verbose == 1:
            print(f"Now processing {video_file_name}")

        video_id = spl[0]
        video_yt_id = "_".join(spl[1:])

        if not use_ffmpeg:
            print("Non-ffmpeg: Not implemented.")
            return None
            # clip = VideoFileClip(video_file_path)
            # duration = clip.duration
            # clip_max_start_time = duration - clip_length
            # clip_starts = np.random.rand(n_clips) * clip_max_start_time
            #
            # self.clips[video_file_name] = []
            #
            # for i in range(n_clips):
            #     if verbose == 1:
            #         print(f"Clip {i + 1} / {n_clips}")
            #     start = clip_starts[i]
            #     end = start + clip_length
            #     subclip_video = clip.subclip(start, end)
            #     self.clips[video_file_name].append(subclip_video)
            #     # subclip_video.write_videofile(env["temp_path"] + "moro.mp4")
        else:
            output_dir = str(Path(f"{self.env['temp_path']}"
                                  f"{target_dataset}/").resolve())

            # The latest point in the video that the random clip can be started
            # from.
            max_start = self.video_duration - clip_length

            # Generate random clip starting points from the range of the video
            # duration.
            clip_starts = np.random.rand(n_clips) * max_start

            if write_clips:
                # Create clips with ffmpeg.
                for i in range(n_clips):
                    ffmpeg_process = subprocess.run([
                        self.env["ffmpeg_location"],  # ffmpeg executable
                        "-ss", str(clip_starts[i]),  # Starting point
                        "-avoid_negative_ts", "1",  # Try to avoid artifacts (TODO)
                        "-i", self.current_file,  # Input file
                        "-c", "copy",  # Copy the file instead of editing it
                        "-t", str(clip_length),  # The duration of the clip
                        f"{output_dir}/{video_file_name}_clip{i}"  # Output file
                    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            return clip_starts

    def load(self, file: str):
        """

        Args:
            file:

        Returns:

        """
        self.current_file = file
        # Get the duration of the video using ffprobe and python subprocess.
        duration_probe = subprocess.run([
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.current_file
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        self.video_duration = float(duration_probe.stdout)

    def get_random_clips_as_list(self,
                                 n_clips: int,
                                 clip_length: float,
                                 sort: Optional[bool] = True):
        """

        Args:
            n_clips:
            clip_length:
            sort:

        Returns:

        """
        self.check_me()
        starts = np.random.rand(n_clips) * self.video_duration
        if sort:
            starts = np.sort(starts)
        clips = []
        for s in starts:
            clips.append((s, s + clip_length))
        return clips

    def get_video_duration(self):
        """

        Returns:

        """
        return self.video_duration

    def create_clip(self,
                    save_path: str,
                    file_name: str,
                    start_t: float,
                    end_t: float):
        """

        Args:
            save_path:
            file_name:
            start_t:
            end_t:

        Returns:

        """

        clip_length = end_t - start_t

        ffmpeg_process = subprocess.run([
            self.env["ffmpeg_location"],  # ffmpeg executable
            "-ss", str(start_t),  # Starting point
            "-avoid_negative_ts", "1",  # Try to avoid artifacts (TODO)
            "-i", self.current_file,  # Input file
            "-c", "copy",  # Copy the file instead of editing it
            "-t", str(clip_length),  # The duration of the clip
            f"{save_path}/{file_name}.mp4"  # Output file
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def check_me(self):
        """

        Returns:

        """
        assert self.video_duration is not None, "Video duration is None"
        assert self.current_file is not None, "Current file is None"

