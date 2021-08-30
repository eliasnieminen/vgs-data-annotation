# Data Annotation

This repository contains tools for data annotation for CrossTask and YouCookII datasets.

All the source code is included in the `src` folder.

## Installation

Use `conda` to create a new environment from `data-annotation.yml`. This will install all the dependencies necessary for the code to run. You may change the environment name from the yml file.

`conda env create -f data-annotation.yml`

Note: On Linux, after the environment has been installed, run the following commands to install `ffmpeg`:

`conda install -c forge ffmpeg`
`conda update ffmpeg`

Note: In order to use YAMNet in the audio segmentation, one needs to use Tensorflow > `2.5.0`. I have not tried to use Tensorflow 1 with YAMNet, but it may work.

## Useful Code

### `src/utilities/construction_utilities.py`
- `get_video_metadata`
  - Returns a metadata object containing useful data about the video
- `get_random_clips_as_list, get_random_clips_as_list_v2`
  - The first one generates random clip lists based on the input video (in the form of `(start, end)` tuples)
  - The second one also generates random clips, but analyzes the contents of each of the clip with the `YamNetSpeechNoiseAnalyzer`.
- `select_sharpest_frames`
  - Is a great example of the usage of the OpenCV's video capture interface, which is used to get the frames from the clips.

### `src/utilities/util.py`
Contains many useful functions

### `src/utilities/clipper.py`
A class for clipping video files. The `write_clip` function shows how to use the Python subprocess interface to write a clip with `ffpmpeg`.

### `src/utilities/env.py`
Contains the `ProjectEnvironment` class for managing paths in the environment. It is used throughout the project.

### `src/data-annotation/annotation_parsing.py`
This file contains a pipeline for constructing structured pickle files from the annotation files provided by the YouCookII and CrossTask dataset authors. The goal is to use one unified way of saving the annotations for both of the datasets.

### `src/training-data-construction/construction_pipeline.py`
This file contains a pipeline for constructing training data for VGS models. One can use random clipping (with audio content analysis) from the dataset videos or use the annotations computed with the `annotation_parsing.py` script.
