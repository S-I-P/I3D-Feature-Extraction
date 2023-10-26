# Copyright 2017 Google Inc.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf

import i3d

#_IMAGE_SIZE = 224
NUM_CLASSES = 400

import os
import argparse
import json

from rgb import rgb_instances

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
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')

    args = parser.parse_args()

    variable_scope = 'RGB'
    ckpt = 'data/checkpoints/rgb_imagenet/model.ckpt'
    num_channel = 3
    statusFile = 'rgb.txt'

    rootdir = args.input_dir
    savedir = args.output
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    duration = {}
    with open(args.input_drn, 'r') as f:
        duration = json.load(f)
    inputFiles = duration.keys()

    done = []
    if os.path.exists(statusFile):
        with open(statusFile, 'r') as f:
            done = f.read().splitlines()
    print(done)
    outfile = open(statusFile, 'a')
    if len(done):
        outfile.write('\n')

    batch_size = args.batch_size

    with open(args.config, 'r') as f:
        config=json.load(f)

    image_height = config['height']
    image_width = config['width']
    seq_length = config["instance_size_seconds"] * config["fps"]

    print("----------")
    print("Feature Extraction: ", variable_scope)
    print("Instance size in frames=", seq_length)
    print("Input shape: #frames", image_width, image_height, num_channel)
    print("----------")

    tf.reset_default_graph()
    with tf.variable_scope(variable_scope):
        input_ = tf.placeholder(tf.float32, [batch_size, seq_length, image_width, image_height, num_channel])
        y_ = tf.placeholder(tf.float32, [batch_size, NUM_CLASSES])
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
                
                saveAt = os.path.join(savedir, file)
                if not os.path.isdir(saveAt):
                    os.mkdir(saveAt)

                print('---------------')
                print(file)
                instances = rgb_instances(os.path.join(root, file), float(duration[file]), config)
                gth_label = np.zeros(shape=[batch_size, NUM_CLASSES])
                total = len(instances)
                features = []
                for i in range(0, total, batch_size):
                    gth_label[0] = 1
                    sample = np.stack(instances[i:i+batch_size], axis=0)
                    sample_len = sample.shape[0]
                    if sample_len<batch_size:
                        extra = np.stack(instances[0:batch_size-sample_len], axis=0)
                        sample = np.concatenate((sample, extra), axis=0)
                    feed_dict = {
                        input_: sample,
                        y_: gth_label,
                        lr: lr_s,
                        drop_out_prob: drop_out
                    }
                    
                    logits, net_feature = sess.run([net, end_points], feed_dict)
                    i3d_feats = net_feature[config['layer']]
                    if sample_len<batch_size:
                        i3d_feats = i3d_feats[0:sample_len]
                    #i3d_feats = np.squeeze(i3d_feats, axis=1)
                    #print(i3d_feats.shape)
                    features.append(i3d_feats)
                    
                features = np.concatenate(features, axis=0)
                print(features.shape)
                np.save(os.path.join(saveAt,'rgb.npy'), features)
                outfile.write(file)
                outfile.write("\n")
                outfile.flush()
    outfile.close()