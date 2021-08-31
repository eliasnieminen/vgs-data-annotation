# Data Annotation

This repository contains tools for data annotation for CrossTask and YouCookII datasets.

All the source code is included in the `src` folder.

## Very short and simple version :)

Video reading:
```python
import cv2 as cv

video_path = "path/to/video"

# Set the start and end:
starting_frame = 100
ending_frame = 200

# Open the capture interface:
cap = cv.VideoCapture(video_path)

# Wait for the video loading.
while not cap.isOpened():
  cap = cv.VideoCapture(video_path)
  print("Waiting...")

# Set starting point:
cap.set(cv.CAP_PROP_POS_FRAMES, starting_frame)

# Set the current position:
pos_frame = starting_frame

frame_counter = 0
# Go through the frames:
while True:
  frame_reading_successful, frame = cap.read()
  if frame_reading_successful:
    # Get the current position:
    pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
    frame_counter += 1
    # Do what ever
    # ====
    # ====
    # Reach the end:
    if frame_counter == ending_frame - starting_frame:
      break
  else:
    # Set cursor back by one.
    cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame - 1)
    # Wait.
    cv.waitKey(250)
    
```

Read audio from video file:

```python
import librosa
video_path = "video/path"
audio, sr = librosa.load(video_path, sr=None)

# There you go!
```

Get blurriness:

```python
import cv2 as cv
image = ... # an np.array
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurriness = cv.Laplacian(image_gray, cv.CV_64F).var()

# There you go!
```

YAMNet:

```python
import src.models.yamnet.yamnet as yamnet_model
import src.models.yamnet.params as yamnet_params

# Prepare YAMNet:

params = yamnet_params.Params(sample_rate=16000, patch_hop_seconds=0.5)
class_names = yamnet_model.class_names("path/to/yamnet_class_map.csv")
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights("path/to/yamnet_weights.h5")

# Load audio (it has to be resampled to 16000 Hz because YAMNet).
video_path = "video/path"
audio, sr = librosa.load(video_path,
                         sr=16000,
                         mono=True,
                         offset=...,  # time of clip start
                         duration=...,  # clip duration
                         res_type="kaiser_fast")

# Use YAMNet:
scores, embeddings, spectrogram = self.yamnet(audio)

# Max scores: returns the max scoring class for each segment.
max_scores = np.argmax(scores, axis=1)

# Suitable class indexes for speech: 0, 1, 2, 3
for class_index in max_scores:
  if class_index in [0, 1, 2, 3]:
    print("Speech! :)")
  else:
    print("Noise! :(")

# All done!
```

Get video duration, the EASY way (but might not be always right)

```python
import librosa
video_path = "video/path"
audio, sr = librosa.load(video_path, sr=None)
duration = librosa.get_duration(audio, sr)
```

Get video duration, the HARD and HACKY but TRUSTWORTHY way:

```python
import subprocess

video_path = "video/path"

duration_probe = subprocess.run(" ".join([
    "ffprobe",
    "-v", "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    video_path
]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

duration = float(duration_probe.stdout)
```

Write video clips (a bit hacky and will contain artifacts):

```python
import subprocess

source_video_path = "path/to/source.mp4"

output_file_name = "video.mp4"
output_directory = "output/dir"

start_time = 10.0
end_time = 20.0

clip_length = end_time - start_time

subprocess.run(" ".join([
  "ffmpeg",  # ffmpeg executable
  "-ss", str(start_time),  # Starting point
  "-avoid_negative_ts", "1",  # Try to avoid artifacts (TODO)
  "-i", source_video_path,  # Input file
  "-c", "copy",  # Copy the file instead of editing it
  "-t", str(clip_length),  # The duration of the clip
  output_directory + output_file_name  # Output file
]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
```

## Installation

Use `conda` to create a new environment. Dependencies:

```
Conda:
python=3.9.5
tensorflow=2.5.0
numpy
scipy

Pip:
matplotlib
opencv-python
scikit-learn
```

Note: On Linux, after the environment has been installed, run the following commands to install `ffmpeg`:

`conda install -c forge ffmpeg`
`conda update ffmpeg`

Note: In order to use YAMNet in the audio segmentation, one needs to use Tensorflow > `2.5.0`. I have not tried to use Tensorflow 1 with YAMNet, but it may work.

### YAMNet Installation

First, clone the Tensorflow models repository: https://github.com/tensorflow/models

Then, go to `models/research/audioset/` and copy the `yamnet` directory to your project directory under `src/models/` directory (create the `src/models/` directory if it doesn't exist).

Then, go to the `yamnet` directory with the command line and run `curl -O https://storage.googleapis.com/audioset/yamnet.h5`. This will download the weights for the YAMNet model.

After this, you might have to edit the source code a bit:

- In `inference.py`:
  - Change `import params as yamnet_params` to `from . import params as yamnet_params`.
  - Change `import yamnet as yamnet_models` to `from . import yamnet as yamnet_models`.
- In `yamnet.py`:
  - Change `import features as features_lib` to `from . import features as features_lib`

## Useful Code

### Speech Detection: `src/analysis/speech_noise_analyser.py`
This file contains the `YamNetSpeechNoiseAnalyser` class which can be used to analyse the contents of an audio from a video file. The class uses `librosa` to load the audio, and `librosa` can also read the audio directly from a video file, so there is no need for additional steps to separate the audio from the video.

Example use for analyzing a whole video:
```python

from src.analysis.speech_noise_analyzer import YamNetSpeechNoiseAnalyzer

sn_analyzer = YamNetSpeechNoiseAnalyser()

video_path = "path/to/.mp4/or/.mkv/or/.webm/video"

# Analyze the whole audio from the video:
speech_proportion, noise_proportion = sn_analyzer.analyze(video_path, whole_video=True)

# If one wants to only analyze a segment from the video:
# (here, a 10 sec clip from 00:00:10.0 to 00:00:20.0 in the video)
speech_proportion_segment, noise_proportion_segment = sn_analyzer.analyze(video_path, start_t=10.0, end_t=20.0)

# This returns all of the segments with speech or noise label
# segments = [
#   {"segment": {start, end} (as floats), "content": 0 = noise or 1 = speech}
# ]
segments = sn_analyzer.get_segments()

# If one wants to plot the segments with matplotlib:
import matplotlib.pyplot as plt
contents = []

# Get the segment contents with a loop.
for segment in segments:
  contents.append(segment["content"])

# Create n for plotting the segments:
n = range(len(contents))

# Plot.
plt.step(n, contents)
plt.show()

```

### Reading Video Data and Generating Random Clips `src/utilities/construction_utilities.py`
- `get_video_metadata`
  - Returns a metadata object containing useful data about the video
- `select_sharpest_frames`
  - Is a great example of the usage of the OpenCV's video capture interface, which is used to get the frames from the clips. `construction_utilities.py -> save_frame` saves the frames to the file system.
- `get_random_clips_as_list, get_random_clips_as_list_v2`
  - The first one generates random clip lists based on the input video (in the form of `(start, end)` tuples)
  - The second one also generates random clips, but analyzes the contents of each of the clip with the `YamNetSpeechNoiseAnalyzer`.

Metadata example:
```python
import matplotlib.pyplot as plt
from src.utilities.construction_utilities.py import get_video_metadata, select_sharpest_frames

video_path = "path/to/.mp4/or/.mkv/or/.webm/file"

# Metadata example:
video_metadata = get_video_metadata(video_path, original_dataset="e.g. youcook2")

# Print some properties:
print(video_metadata.fps, video_metadata.duration, video_metadata.metadata)

# Other properties available from the metadata dict:
yt_id = video_metadata.metadata["yt_id"]
task_id = video_metadata.metadata["task_id"]
file_name = video_metadata.metadata["file_name"]
original_dataset = video_metadata.metadata["original_dataset"]

```

Sharpest frames example (continuation of the previous example):

```python
start_t = 10.0
end_t = 20.0

# Convert times to frames with the help of the fps property of the video metadata:
start_f = int(start_t * video_metadata.fps)
end_f = int(end_t * video_metadata.fps)

# Find the best frames for each second of the clip (start_f, end_f):
# return_type = "path" makes the algorithm to save the best frames to file system and return a list of paths
# instead of a list of numpy arrays containing the uncompressed images.
best_frames = select_sharpest_frames(video_path, 
                                     start_f, 
                                     end_f, 
                                     video_metadata,
                                     return_type="path",  
                                     save_dir="your/image/directory")

# If one wants to plot a frame:
image = plt.imread(best_frames[0])
plt.imshow(image)
plt.show()
```

Random clip example (using the `get_random_clips_as_list_v2` function of the random clip generation):

```python
from src.utilities.construction_utilities import get_random_clips_as_list_v2

# This will return n_clips amount of random clips from the span of the video_metadata.duration
# as two lists of tuples: one with clip information as seconds, one with clip information as frames.
# If use_speech_detection is True, the clips will be analyzed by the YamNetSpeechNoiseAnalyzer and
# the snr_threshold parameter defines which clips will be accepted.
clip_list, clip_list_as_frames = get_random_clips_as_list_v2(
  video_metadata=video_metadata,
  n_clips=10,
  clip_length=15.0,  # seconds
  use_speech_detection=False,
  snr_threshold=0.8)
```

### Video, Audio and Image Writing

For video writing, `ffmpeg` is used. `src/utilities/clipper.py` contains a class for clipping video files (`Clipper`). The `Clipper.write_clip` function shows how to use the Python subprocess interface to write a clip with `ffpmpeg`. There is also an API for video writing provided by OpenCV, but it was not used here due to speed issues. When clipping the videos, some artifacts will be seen in the beginning of the clips, this is due to the video codec.

For image writing, `opencv`'s `cv2.imwrite` function is used. See example in `src/utilities/construction_utilities.py -> save_frame`. _Keep in mind, that OpenCV uses BRG color model, so there might be some confusion when dealing with color images._

For audio writing, `soundfile` package is used: `soundfile.write(save_path, data, sampling_rate)`. See example in `src/training-data-construction/construction_pipeline.py -> row 237`.

### Blur Detection

`src/analysis/blur_detection.py` contains a class `BlurDetector`.

Usage example:
```python

from src.analysis.blur_detection import BlurDetector

img_path = "path/to/.jpg/or/.png/image"  # Other formats might be supported but not tried.

blur_detector = BlurDetector()
blur_detector.detect(img_path)
blurriness = blur_detector.get_blurriness()

print(blurriness)

# The image could also be a np.ndarray:
image = plt.imread(img_path)
blur_detector.detect(image)
blurriness = blur_detector.get_blurriness()

print(blurriness)
```

Since there is no absolute value to determine if the image is blurry or not, some threshold should be defined for the blurriness. The smaller the value is, the blurrier the image is.

### `src/utilities/clipper.py`
Contains a class for clipping video files (`Clipper`). The `Clipper.write_clip` function shows how to use the Python subprocess interface to write a clip with `ffpmpeg`.

### `src/utilities/util.py`
Contains many useful functions

### `src/utilities/env.py`
Contains the `ProjectEnvironment` class for managing paths in the environment. It is used throughout the project.

### `src/data-annotation/annotation_parsing.py`
This file contains a pipeline for constructing structured pickle files from the annotation files provided by the YouCookII and CrossTask dataset authors. The goal is to use one unified way of saving the annotations for both of the datasets.

### `src/training-data-construction/construction_pipeline.py`
This file contains a pipeline for constructing training data for VGS models. One can use random clipping (with audio content analysis) from the dataset videos or use the annotations computed with the `annotation_parsing.py` script.
