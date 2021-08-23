import librosa as lbr
import numpy as np
from typing import Optional, Union, List, Dict

import models.yamnet.yamnet as yamnet_model
import models.yamnet.params as yamnet_params


class YamNetSpeechNoiseAnalyzer:
    def __init__(self,
                 sr: Optional[float] = 16000.0,
                 hop: Optional[float] = 0.5,
                 res_type: Optional[str] = "kaiser_fast"):
        self.sr = sr
        self.hop = hop
        self.res_type = res_type

        self.params = yamnet_params.Params(sample_rate=self.sr,
                                           patch_hop_seconds=self.hop)
        self.class_names = yamnet_model.class_names(
            "models/yamnet/yamnet_class_map.csv")
        self.yamnet = yamnet_model.yamnet_frames_model(self.params)
        self.yamnet.load_weights("models/yamnet/yamnet.h5")

    def analyze(self, video_path, start_t, end_t) -> tuple:
        clip_dur = end_t - start_t
        audio, sr = lbr.load(video_path,
                             sr=self.sr,
                             mono=True,
                             offset=start_t,
                             duration=clip_dur,
                             res_type=self.res_type)

        audio_len_s = len(audio) / sr

        scores, embeds, spec = self.yamnet(audio)
        scores = scores.numpy()

        scores_max = np.argmax(scores, axis=1)

        scores_max = list(np.ndarray.tolist(scores_max))
        speech_count = 0
        noise_count = 0
        count = 0
        segments = []
        for class_index in scores_max:
            start_pos = count * self.hop
            end_pos = (count + 1) * self.hop
            seg = {
                "segment": (start_pos, end_pos),
            }
            if class_index == 0 or class_index == 1:
                speech_count += 1
                seg["content"] = 0
            else:
                noise_count += 1
                seg["content"] = 1
            segments.append(seg)
            count += 1

        speech_proportion = speech_count / len(scores_max)
        noise_proportion = noise_count / len(scores_max)

        print(f"Speech proportion: "
              f"{'{:.2f}'.format(speech_proportion)}")
        print(f"Noise proportion: "
              f"{'{:.2f}'.format(noise_proportion)}")

        return speech_proportion, noise_proportion

