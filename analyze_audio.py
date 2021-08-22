import pickle

from speech_noise_analyzer import SpeechNoiseAnalyzer
from pathlib import Path
from env import ProjectEnvironment
import psutil

print("Yeah")

env = ProjectEnvironment()

target_dataset = "youcook2"

p = Path(f"download/{target_dataset}").resolve()

ram_usage = []
ram_usage_dir = Path("misc/").resolve()

for file in p.iterdir():
    if file.suffix == ".mp4" or file.suffix == ".mkv" or file.suffix == ".webm":
        spl = file.stem.split(".")[0].split("_")
        video_id = spl[0]
        yt_id = "_".join(spl[1:])
        metadata_filename = f"{video_id}_{yt_id}.pickle"

        print(f"Now processing {metadata_filename.strip('.pickle')}...")

        if not Path(f"metadata/{target_dataset}/{metadata_filename}").resolve().exists():
            try:
                # Report RAM usage.
                ram_usage.append(psutil.virtual_memory().percent)
                with open(str(Path(ram_usage_dir / "ram_usage.pickle").resolve()), 'wb') as ram_usage_file:
                    pickle.dump(ram_usage, ram_usage_file)

                # Analyze file.
                analyzer = SpeechNoiseAnalyzer(precalc=False,
                                               save_precalc=True,
                                               save_metadata=True,
                                               draw_plot=True,
                                               ffmpeg=env["ffmpeg_location"],
                                               delete_segmenter=True)
                analyzer.analyze(str(file))
                del analyzer
            except AssertionError:
                print("Assertion Error, continuing to next video")
                continue
        else:
            print("Skipping, file already exists.")
