import os
import numpy as np
from sys import argv

if len(argv)<2:
    print("Usage: python flow_feature_combine.py input_dir")
    exit(1)

rootdir = argv[1]

for root, _, files in os.walk(rootdir):
    if "flow.npy" in files and "flow1st.npy" in files:
        flow1 = np.load(os.path.join(root, 'flow1st.npy'))
        flow_ = np.load(os.path.join(root, 'flow.npy'))
        flow = np.concatenate((flow1, flow_), axis=0)
        np.save(os.path.join(root, 'flow.npy'), flow)
        os.remove(os.path.join(root, 'flow1st.npy'))