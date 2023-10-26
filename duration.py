import os
import subprocess
from sys import argv
import json

def getDuration(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

if len(argv)<4:
    print("Usage: python duration.py input_dir input_files_list output_file_name")
    exit(1)

fname = argv[2]
with open(fname, 'r') as f:
    inputList = f.read().splitlines()

durations = {}

for root,_,files in os.walk(argv[1]):
    for file in files:
        if file in inputList:
            fpath = os.path.join(root, file)
            duration = getDuration(fpath)
            print(fpath, duration)
            durations[file] = str(duration)

with open(argv[3], 'w') as f:
    json.dump(durations, f)