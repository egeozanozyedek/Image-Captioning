from pathlib import Path



def define_path(path_name):
    Path(path_name).mkdir(parents=True, exist_ok=True)