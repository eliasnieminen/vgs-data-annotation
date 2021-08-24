import numpy as np

from pathlib import Path
from env import ProjectEnvironment


env = ProjectEnvironment()

# Select 100 videos (randomly)

# Do clipping based on annotations
# Do clipping based on Yamnet analysis

target_dataset = "youcook2"
target_split = "train"
override_save_patj