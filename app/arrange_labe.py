import os
import re

path ="/home/mrcloud/Downloads/manga-translator-detection.v4-2000.yolov8/train/labels/"

all_lables = os.listdir(path)
print(all_lables)
pattern = '-\d*-'
filtered_strings = [string for string in all_lables if re.search(pattern, string)]

print(filtered_strings )