from typing import Optional, Union, Dict, List

import numpy as np
from pathlib import Path
import subprocess
from src.utilities.env import ProjectEnvironment


class Clipper:
    """Class for clipping videos.

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
        """Create random clips from a video.

        Args:
            n_clips: How many clips?
            clip_length: How long clips?
            target_dataset: From which dataset?
            verbose: Print logs?
            use_ffmpeg: There is no other option than True at the moment.
            write_clips: Write the clips to file?

        Returns: Clip onsets.

        """

        self.check_me()

        video_file_name = Path(self.current_file).resolve().stem

        if verbose == 1:
            print(f"Now processing {video_file_name}")

        if not use_ffmpeg:
            print("Non-ffmpeg: Not implemented.")
            return None
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
        """Loads a video (not to memory) and gets some metadata as well.

        Args:
            file: The file to be loaded.

        Returns: None

        """
        self.current_file = file
        # Get the duration of the video using ffprobe and python subprocess.
        duration_probe = subprocess.run(" ".join([
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.current_file
        ]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

        self.video_duration = float(duration_probe.stdout)

    def get_video_duration(self):
        """Gets the video duration.

        Returns: Video duration as float.

        """
        return self.video_duration

    def create_clip(self,
                    save_path: str,
                    file_name: str,
                    start_t: float,
                    end_t: float):
        """Writes a clip to file.

        Args:
            save_path: The destination directory of the clip.
            file_name: The file name of the clip.
            start_t: The starting time of the clip.
            end_t: The ending time of the clip.

        Returns: None

        """

        clip_length = end_t - start_t

        ffmpeg_process = subprocess.run(" ".join([
            self.env["ffmpeg_location"],  # ffmpeg executable
            "-ss", str(start_t),  # Starting point
            "-avoid_negative_ts", "1",  # Try to avoid artifacts (TODO)
            "-i", self.current_file,  # Input file
            "-c", "copy",  # Copy the file instead of editing it
            "-t", str(clip_length),  # The duration of the clip
            f"{save_path}/{file_name}.mp4"  # Output file
        ]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    def check_me(self):
        """Checks if everything is ok.

        Returns: None

        """
        assert self.video_duration is not None, "Video duration is None"
        assert self.current_file is not None, "Current file is None"

