import os
import sys


def _init_path():
    cur_path = "."
    main_path = ".."
    for path in [cur_path, main_path]:
        sys.path.append(os.path.abspath(path))


_init_path()
