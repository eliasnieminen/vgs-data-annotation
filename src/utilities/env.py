from pathlib import Path


class ProjectEnvironment:
    """This class handles the paths of each environment and platform.

    If one wants to add their own environment, id can be added to the
    self.env dictionary. The mechanism for selecting the environment can be
    defined in the self.determine_env-function.

    """
    def __init__(self):
        self.env_id = self.determine_env()
        self.env = {
            0: {
                "base_path": "c:/users/elias/pycharmprojects/data-annotation/",
                "data_path": "data/",
                "download_path": "download/",
                "metadata_path": "metadata/",
                "temp_path": "_temp/",
                "clip_save_path": "clips/",
                "clip_annotations_save_path": "clip_annotations/",
                "precalc_segments_path": "segments_precalc",

                "crosstask_path": "data/crosstask_release/",
                "youcook2_path": "data/youcook2/",
                "howto100m_path": "data/howto100m/",

                # Training data save paths
                "crosstask_save_path": "download/crosstask/train/",
                "youcook2_save_path": "download/youcook2/train/",
                "howto100m_save_path": "download/howto100m/train/",

                # Test data save paths (only for youcook2)
                "youcook2_save_path_test": "download/youcook2/test/",

                # Validation data save paths (only for crosstask and youcook2)
                "crosstask_save_path_val": "download/crosstask/val/",
                "youcook2_save_path_val": "download/youcook2/val/",

                "ffmpeg_location": "c:/tools/ffmpeg/bin/ffmpeg.exe",
                "yt_dl_executable": "c:/tools/youtube-dl/youtube-dl.exe"
            },
            1: {
                "base_path": str(Path("../..").resolve()) + "/",
                "data_path": "data/",
                "download_path": "download/",
                "metadata_path": "metadata/",
                "temp_path": "_temp/",
                "clip_save_path": "clips/",
                "clip_annotations_save_path": "clip_annotations/",
                "precalc_segments_path": "segments_precalc",

                "crosstask_path": "data/crosstask_release/",
                "youcook2_path": "data/youcook2/",
                "howto100m_path": "data/howto100m/",

                # Training data save paths
                "crosstask_save_path": "download/crosstask/train/",
                "youcook2_save_path": "download/youcook2/train/",
                "howto100m_save_path": "download/howto100m/train/",

                # Test data save paths (only for youcook2)
                "youcook2_save_path_test": "download/youcook2/test/",

                # Validation data save paths (only for crosstask and youcook2)
                "crosstask_save_path_val": "download/crosstask/val/",
                "youcook2_save_path_val": "download/youcook2/val/",

                "ffmpeg_location": "ffmpeg",
                "yt_dl_executable": "youtube-dl"
            }
        }

    @staticmethod
    def determine_env() -> int:
        """This function determines the environment id which is used to select
                the appropriate paths depending on the platform the script is excecuted.
                Here you can also add your own environment selection mechanism.

        Returns: The environment id.

        """
        cd = Path("../..").resolve()
        if "C:" in str(cd):
            return 0
        elif "lustre" in str(cd):
            return 1
        else:
            return 0

    def __getitem__(self, item):
        return self.env[self.env_id][item]

    def create_directories(self):
        """Tries to automate the creation of all the necessary directories.

        Probably won't work at this moment...

        Returns: None

        """
        env = self.env[self.env_id]
        for path in env.keys():
            if "path" in path:
                p = Path(env[path]).resolve()
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
