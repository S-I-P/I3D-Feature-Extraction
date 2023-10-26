import cv2
import numpy as np

def rgb_instances(file, duration, config):
    print(duration)
    cap = cv2.VideoCapture(file)
    frames = []
    height = config['height']
    width = config['width']
    fps = config['fps']
    inst_sz = config['instance_size_seconds']
    stride = config['stride']
    seq_length = int(inst_sz*fps)
    
    while True:
        ret, frame = cap.read()
        if not(ret):
            break
        frame = cv2.resize(frame, (width, height))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    cv2.destroyAllWindows()
    print(len(frames))

    start_time = 0.0
    instances = []
    idx1 = 0
    idx2 = 0
    
    while start_time<duration:
        end_time = start_time + inst_sz
        idx1 = round(start_time*fps)
        if idx1>=len(frames):
            break
        idx2 = round(end_time*fps)
        rgb_instance = frames[idx1:idx2]
        rgb_instance = np.stack(rgb_instance, axis=0)
        rgb_instance = (rgb_instance*2.0/255.0) - 1.0
        instances.append(rgb_instance)
        start_time = start_time + stride * inst_sz
        if idx2>=len(frames):
            break

    lastShape = instances[-1].shape[0]
    if lastShape<seq_length:
        #print(lastShape)
        pad = np.zeros((seq_length-lastShape, height, width, 3))
        last = np.concatenate((instances[-1], pad), axis=0)
        instances[-1] = last

    return instances