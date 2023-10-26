import cv2
import numpy as np
import threading
from operator import itemgetter
from math import ceil

flows_ = []

class thread(threading.Thread):
    def __init__(self, i, instance):
        threading.Thread.__init__(self)
        self.i = i
        self.instance = instance
        self.tvl1 = cv2.createOptFlow_DualTVL1()
    def run(self):
        thFlows = []
        for ins in self.instance:
            prv = cv2.cvtColor(ins[0], cv2.COLOR_BGR2GRAY)
            flow_instance = []
            for frame in ins[1:]:
                nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = self.tvl1.calc(prv, nxt, None)
                flow_instance.append(flow)
                prv = nxt
            flow_instance = np.stack(flow_instance, axis=0)
            flow_instance = np.clip(flow_instance, -20, 20)
            flow_instance = flow_instance/20
            thFlows.append(flow_instance)
        flows_.append((self.i, thFlows))

def flow_instances(file, duration, config, numThreads):
    global flows_
    flows_ = []
    cap = cv2.VideoCapture(file)
    frames = []
    height = config['height']
    width = config['width']
    fps = config['fps']
    inst_sz = config['instance_size_seconds']
    stride = config['stride']
    seq_length = int(inst_sz*fps)
    print(duration)
    while True:
        ret, frame = cap.read()
        if not(ret):
            break
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    print(len(frames))
    cap.release()
    cv2.destroyAllWindows()

    start_time = 0.0
    end_time = start_time + inst_sz
    instances = []
    idx1 = round(start_time*fps)
    idx2 = round(end_time*fps)
    temp_instance = frames[idx1:idx2]
    prev = temp_instance[-1]
    start_time = start_time + stride * inst_sz

    while start_time<duration:
        temp_instance = []
        end_time = start_time + inst_sz
        idx1 = round(start_time*fps)
        if idx1>=len(frames):
            break
        idx2 = round(end_time*fps)
        temp_instance = frames[idx1:idx2]
        temp_instance = [prev] + temp_instance
        prev = temp_instance[-1]
        instances.append(temp_instance)
        start_time = start_time + stride * inst_sz
        if idx2>=len(frames):
            break

    total = len(instances)
    threadSize = ceil(total/numThreads)
    threads = []
    idx1 = 0
    
    for i in range(numThreads):
        idx2 = idx1 + threadSize
        t = thread(idx1, instances[idx1:idx2])
        t.start()
        threads.append(t)
        idx1 = idx2

    for t in threads:
        t.join()

    flowsTemp = [x for _,x in list(sorted(flows_, key =itemgetter(0)))]
    flows = []
    for fl in flowsTemp:
        for f_ in fl:
            flows.append(f_)
    
    lastShape = flows[-1].shape[0]
    if lastShape<seq_length:
        last = flows[-1].copy()
        pad = np.zeros((seq_length-lastShape, height, width, 2))
        last = np.concatenate((last, pad), axis=0)
        flows[-1] = last

    return flows