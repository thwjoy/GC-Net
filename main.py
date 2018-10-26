import argparse
import os
# import scipy.misc
import numpy as np
import tensorflow as tf
from model import network

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_list_dir_train', dest='data_list_dir_train', default='', help='path of the training dataset')
parser.add_argument('--data_list_dir_test', dest='data_list_dir_test', default='', help='path of the testing dataset')
parser.add_argument('--phase', dest='phase', help='train, test', required=True, choices=['train','test'])

parser.add_argument('--epoch', dest='epoch', type=int, default=5, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512, help='# images in batch')
parser.add_argument('--image_dim', dest='image_size', type=int, default=, help='Image sizes')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--checkpoint', dest='checkpoint_dir', default='./data/ckpt', help='path to checkpoint folber for weights and tensorboard')
parser.add_argument('--dataset', dest='dataset_name', default='', help='dataset for which the weights are to be trained')


args = parser.parse_args()

def main(_):
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    with tf.Session(config=config) as sess:
        model = network(sess, args)
        if args.phase == 'train':
            model.train(args)
        else: 
            model.test(args)

if __name__ == '__main__':
    tf.app.run()
