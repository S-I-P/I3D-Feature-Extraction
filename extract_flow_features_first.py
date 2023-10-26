from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2

import i3d

#_IMAGE_SIZE = 224
NUM_CLASSES = 400

import os
import argparse
import json

tvl1 = cv2.createOptFlow_DualTVL1()
# flow instance first
def flow_instance(file, num_frames, width, height):
    cap = cv2.VideoCapture(file)    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not(ret):
            break
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
        if len(frames)==num_frames:
            break
    cap.release()
    cv2.destroyAllWindows()

    flows = []
    prv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = tvl1.calc(prv, nxt, None)
        flows.append(flow)
        prv = nxt
            
    flows = np.stack(flows, axis=0)
    flows = np.clip(flows, -20, 20)
    flows = flows/20

    return flows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_drn', default=None, type=str,
                            help='Input Files Duration')
    parser.add_argument('--input_dir', default='../Videos', type=str,
                            help='Directory containing videos')
    parser.add_argument('--config', default='config.json', type=str,
                            help='configuration file')
    parser.add_argument('--output',default='features', type=str,
                            help='Destination folder to save features')

    args = parser.parse_args()

    variable_scope = 'Flow'
    ckpt = 'data/checkpoints/flow_imagenet/model.ckpt'
    num_channel = 2
    statusFile = 'flow1st.txt'

    rootdir = args.input_dir
    savedir = args.output
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    with open(args.input_drn, 'r') as f:
        inputFiles = json.load(f).keys()

    done = []
    if os.path.exists(statusFile):
        with open(statusFile, 'r') as f:
            done = f.read().splitlines()
    print(done)
    outfile = open(statusFile, 'a')
    if len(done):
        outfile.write('\n')

    with open(args.config, 'r') as f:
        config=json.load(f)

    image_height = config['height']
    image_width = config['width']
    seq_length = int(config["instance_size_seconds"] * config["fps"])-1

    print("----------")
    print("Feature Extraction: ", variable_scope)
    print("Instance size in frames=", seq_length)
    print("Input shape: ", seq_length, image_width, image_height, num_channel)
    print("----------")

    tf.reset_default_graph()
    with tf.variable_scope(variable_scope):
        input_ = tf.placeholder(tf.float32, [1, seq_length, image_width, image_height, num_channel])
        y_ = tf.placeholder(tf.float32, [1, NUM_CLASSES])
        lr = tf.placeholder("float")
        drop_out_prob = tf.placeholder("float")
        i3d_model = i3d.InceptionI3d(num_classes=NUM_CLASSES, final_endpoint=config['layer'])
        net, end_points = i3d_model(input_, is_training=False, dropout_keep_prob=drop_out_prob)

    variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == variable_scope:
            variable_map[variable.name.replace(':0', '')] = variable

    tf_config = tf.ConfigProto()
    restorer = tf.train.Saver(var_list=variable_map, reshape=True)
    with tf.Session(config=tf_config) as sess:
        restorer.restore(sess, ckpt)
        lr_s = 0.0001
        drop_out = 1

        for root,_,files in os.walk(rootdir):
            for file in files:
                if file not in inputFiles:
                    continue
                if file in done:
                    continue
                
                instance = flow_instance(os.path.join(root, file), seq_length+1, image_width, image_height)
                saveAt = os.path.join(savedir, file)
                print("---------------------")
                print(file)
                if not os.path.isdir(saveAt):
                    os.mkdir(saveAt)

                sample = np.zeros(shape=[1, seq_length, image_width, image_height, num_channel])
                sample[0,:,:,:,:] = instance
                print(sample.shape)
                gth_label = np.zeros(shape=[1, NUM_CLASSES])
                gth_label[0] = 1
                
                feed_dict = {
                    input_: sample,
                    y_: gth_label,
                    lr: lr_s,
                    drop_out_prob: drop_out
                }
                
                logits, net_feature = sess.run([net, end_points], feed_dict)
                i3d_feats = net_feature[config['layer']]
                #i3d_feats = np.squeeze(i3d_feats, axis=1)
                print(i3d_feats.shape)
                
                np.save(os.path.join(saveAt,'flow1st.npy'), i3d_feats)
                outfile.write(file)
                outfile.write("\n")
                outfile.flush()

    outfile.close()