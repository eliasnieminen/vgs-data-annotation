from pathlib import Path
# from inaSpeechSegmenter import Segmenter
import pickle

import librosa as lbr
import numpy as np
from typing import Optional, Union, List, Dict

import models.yamnet.yamnet as yamnet_model
import models.yamnet.params as yamnet_params


# class SpeechNoiseAnalyzer:
    # def __init__(self,
                 # precalc: Optional[bool] = True,
                 # save_precalc: Optional[bool] = True,
                 # save_metadata: Optional[bool] = True,
                 # draw_plot: Optional[bool] = False,
                 # plt_args: Optional[Union[None, dict]] = None,
                 # ffmpeg: Optional[str] = None,
                 # delete_segmenter: Optional[bool] = False):
        # self.env = ProjectEnvironment()
        # self.precalc = precalc
        # self.save_precalc = save_precalc
        # self.draw_plot = draw_plot
        # self.plt_args = plt_args
        # self.seg = Segmenter(ffmpeg=ffmpeg) if not self.precalc else None
        # self.delete_segmenter = delete_segmenter
        # self.save_metadata = save_metadata
        # self.precalc_segments_path = self.env["precalc_segments_path"]
        # self.shares = {
            # "male": 0,
            # "female": 0,
            # "music": 0,
            # "noise": 0,
            # "noEnergy": 0
        # }

    # def analyze(self, file_path: str) -> dict:

        # file = Path(file_path).resolve()

        # file_stem = file.stem
        # original_dataset = file.parent.stem

        # precalc_load_file = f"{self.precalc_segments_path}/" \
                            # f"{original_dataset}/{file_stem}_metadata.pickle"

        # precalc_save_file = f"{self.precalc_segments_path}/" \
                            # f"{original_dataset}/{file_stem}_metadata.pickle"

        # if not self.precalc:
            # segments = self.seg(str(file))
            # if self.delete_segmenter:
                # del self.seg
            # if self.save_precalc:
                # with open(precalc_save_file, 'wb') as pickle_file:
                    # pickle.dump({
                        # "segments": segments,
                        # "original_file": str(file)
                    # }, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        # else:
            # with open(precalc_load_file, 'rb') as pickle_file:
                # loaded = pickle.load(pickle_file)
                # segments = loaded["segments"]

        # audio, sr = lbr.load(str(file), sr=None, mono=True)
        # len_audio_sec = len(audio) / sr

        # labels = {}

        # seg_count = 0
        # for segment in segments:
            # label = segment[0]
            # segment_start = float(segment[1])
            # segment_end = float(segment[2])
            # segment_length = segment_end - segment_start

            # if label not in labels.keys():
                # labels[label] = segment_length
            # else:
                # labels[label] += segment_length

            # offset = (seg_count % 3) * 0.05
            # if self.draw_plot:
                # plt.vlines(segment_start, ymin=-1, ymax=1, color='r')
                # plt.text(x=segment_start,
                         # y=0.95 - offset,
                         # s=f"{label}",
                         # fontsize="large")

            # seg_count += 1

        # if self.draw_plot:
            # n = np.linspace(0, len_audio_sec, num=len(audio))
            # plt.plot(n, audio)
            # plt.xlabel("Time (s)")
            # plt.ylabel("Amplitude")
            # plt.title(file_stem)
            # plt.show()

        # cum_share = 0
        # noise_cum_share = 0
        # speech_cum_share = 0

        # for label in labels.keys():
            # share = 100 * labels[label] / len_audio_sec
            # if label in ["noise", "noEnergy", "music"]:
                # noise_cum_share += share
            # elif label in ["male", "female"]:
                # speech_cum_share += share

            # cum_share += share
            # self.shares[label] = share / 100
            # print(f"{label} share: {share}")

        # self.shares["cum_share"] = cum_share
        # self.shares["noise_cum_share"] = noise_cum_share
        # self.shares["speech_cum_share"] = speech_cum_share

        # if self.save_metadata:
            # self.write_metadata(original_file=file.name,
                                # original_dataset=original_dataset)

        # del audio, sr
        # return self.shares

    # def get_shares(self):
        # return self.shares

    # def write_metadata(self,
                       # original_file: str,
                       # original_dataset: str):
        # metadata = {
            # "file": {
                # "original_file": original_file,
                # "original_dataset": original_dataset
            # },
            # "metadata": {
                # "speech_noise_data": self.shares
            # }
        # }

        # spl = original_file.split(".")[0].split("_")
        # video_id = spl[0]
        # yt_id = "_".join(spl[1:])

        # filename = f"{self.env['metadata_path']}{original_dataset}/" \
                   # f"{video_id}_{yt_id}.pickle"

        # with open(filename, "wb") as file:
            # pickle.dump(metadata, file, protocol=pickle.HIGHEST_PROTOCOL)

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

    def analyze(self,
                video: Union[str, np.ndarray],
                sr: Optional[Union[float, None]] = None,
                start_t: Optional[Union[float, None]] = None,
                end_t: Optional[Union[float, None]] = None,
                whole_video: Optional[bool] = False) -> Union[tuple, None]:

        if not whole_video:
            clip_dur = end_t - start_t
        else:
            start_t = 0
            clip_dur = None

        try:
            audio, sr = lbr.load(video,
                                 sr=self.sr,
                                 mono=True,
                                 offset=start_t,
                                 duration=clip_dur,
                                 res_type=self.res_type)
        except RuntimeError:
            return None

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

