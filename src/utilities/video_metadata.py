import math

from typing import Optional, Union, Dict


class VideoMetadata:
    """Metadata object for videos.

    """
    def __init__(self,
                 dur: Optional[Union[None, float]] = None,
                 fps: Optional[Union[None, float]] = None,
                 metadata: Optional[Union[None, Dict]] = None):
        self.dur = dur
        self.framerate = fps
        self.video_metadata = metadata

    def set_duration(self, duration):
        self.dur = duration

    def set_fps(self, fps):
        self.framerate = fps

    def set_video_metadata(self, metadata):
        self.video_metadata = metadata

    @property
    def duration(self):
        return self.dur

    @property
    def fps(self):
        return self.framerate

    @property
    def metadata(self):
        return self.video_metadata

    @property
    def frame_count(self):
        return math.floor(self.framerate * self.duration)
