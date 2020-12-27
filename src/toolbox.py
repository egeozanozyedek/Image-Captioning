from pathlib import Path
from progress.bar import Bar
import requests
import os
import sys
import time

KEY = 'KEY'
SECRET = 'SECRET'

SIZES = ["url_o", "url_k", "url_h", "url_l", "url_c"]


def estimate_time(start_time, iterations_left):
    time_remain = (time.time() - start_time) * iterations_left
    if time_remain > 3600:
        time_remain = round(time_remain / 3600)
        time_label = "hour(s)"
    elif time_remain > 60:
        time_remain = round(time_remain / 60)
        time_label = "minute(s)"
    else:
        time_remain = round(time_remain)
        time_label = "second(s)"

    return time_remain, time_label, time.time()


def define_path(pathname):
    Path(pathname).mkdir(parents=True, exist_ok=True)

def download_data(data, pathname):

    for url in data:

        name = url.split("/")[-1]
        path = os.path.join(pathname, name)

        if os.path.isfile(pathname) is False:
            response = requests.get(url, stream=True)
            with open(path, 'wb') as outfile:
                outfile.write(response.content)


