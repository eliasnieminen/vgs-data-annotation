# from inaSpeechSegmenter import Segmenter

import librosa as lbr
import numpy as np
from typing import Optional, Union

import src.models.yamnet.yamnet as yamnet_model
import src.models.yamnet.params as yamnet_params
from src.utilities.env import ProjectEnvironment

env = ProjectEnvironment()


class YamNetSpeechNoiseAnalyzer:
    """Analyzes the contents of a given audio file and segment.

    Utilizes the YAMNet model.
    https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
    """
    def __init__(self,
                 sr: Optional[float] = 16000.0,
                 hop: Optional[float] = 0.5,
                 res_type: Optional[str] = "kaiser_fast"):
        """Initialization of the class.

        Args:
            sr: The sample rate that the audio files will be resampled to
                before processing. YAMNet uses sr = 16000 so it is the default
                and the only reasonable value here.
            hop: The segment size of the analysis.
            res_type: The resampling type for librosa. Defaults to kaiser_fast
                      since fast resampling is needed in the processing of
                      multiple files.
        """
        self.sr = sr
        self.hop = hop
        self.res_type = res_type

        # YAMNet parameters.
        self.params = yamnet_params.Params(sample_rate=self.sr,
                                           patch_hop_seconds=self.hop)
        # YAMNet classes.
        self.class_names = yamnet_model.class_names(
            f"{env['base_path']}src/models/yamnet/yamnet_class_map.csv")
        # The model itself.
        self.yamnet = yamnet_model.yamnet_frames_model(self.params)
        # Load weights.
        self.yamnet.load_weights(f"{env['base_path']}src/models/yamnet/yamnet.h5")
        # The analyzed segments container.
        self.segments = []

    def analyze(self,
                video: Union[str, np.ndarray],
                start_t: Optional[Union[float, None]] = None,
                end_t: Optional[Union[float, None]] = None,
                whole_video: Optional[bool] = False) -> Union[tuple, None]:
        """Analyzes the contents of the audio signal and segment provided.

        Args:
            video: The path of the video from which the audio is processed.
            start_t: The starting point of the analysis segment (in sec).
            end_t: The ending point of the analysis segment (in sec).
            whole_video: Flag for analyzing the whole video instead of a
                         specific segment.

        Returns: A tuple containing the ratios for speech and noise.
                 (speech_proportion, noise_proportion)

        """

        if not whole_video:
            clip_dur = end_t - start_t
        else:
            start_t = 0
            clip_dur = None

        try:
            # Load the audio.
            # If there is an error, stop execution and return None.
            audio, sr = lbr.load(video,
                                 sr=self.sr,
                                 mono=True,
                                 offset=start_t,
                                 duration=clip_dur,
                                 res_type=self.res_type)
        except RuntimeError:
            return None

        # Process the audio by feeding it to the YAMNet model.
        scores, embeds, spec = self.yamnet(audio)
        scores = scores.numpy()

        # Get the max-scored class for each segment.
        scores_max = np.argmax(scores, axis=1)

        # Transform to regular list.
        scores_max = list(np.ndarray.tolist(scores_max))

        # Counters for segment contents.
        speech_count = 0
        noise_count = 0
        count = 0

        # Container for individual segments.
        segments = []

        # Go through the obtained classes for each segment.
        for class_index in scores_max:
            # Get the start and end position of this segment.
            start_pos = count * self.hop
            end_pos = (count + 1) * self.hop
            seg = {
                "segment": (start_pos, end_pos),
            }

            # The class labels can be found in the YAMNet's class map in
            # the models/yamnet directory.

            # The segment contains speech.
            if class_index == 0 or \
               class_index == 1 or \
               class_index == 2 or \
               class_index == 3:
                speech_count += 1
                seg["content"] = 1
            # The segment contains noise.
            else:
                noise_count += 1
                seg["content"] = 0
            # Append the segment to the container list.
            segments.append(seg)
            count += 1

        # Calculate the proportions.
        speech_proportion = speech_count / len(scores_max)
        noise_proportion = noise_count / len(scores_max)

        print(f"Speech proportion: "
              f"{'{:.2f}'.format(speech_proportion)}")
        print(f"Noise proportion: "
              f"{'{:.2f}'.format(noise_proportion)}")

        self.segments = segments

        return speech_proportion, noise_proportion

    def get_segments(self):
        return self.segments
