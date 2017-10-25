#!/usr/bin/env python

# Graph the corresponding pixel values in a sequence of images to assert linearity.

from __future__ import division

from argparse import ArgumentParser
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import cv2


arg_parser = ArgumentParser()
arg_parser.add_argument('-k', '--like')
arg_parser.add_argument('-s', '--sample', action='append')
arg_parser.add_argument('-a', '--all-channels', action='store_true')
arg_parser.add_argument('-r', '--reference', type=int, default=-1)
arg_parser.add_argument('-c', '--count', type=int, default=5)
arg_parser.add_argument('-t', '--title')
arg_parser.add_argument('src', nargs=1)
arg_parser.add_argument('dst', nargs='?')
args = arg_parser.parse_args()


#src = args.src[0]

def plot_crc(src):
    max_value = 2.0 ** 16
    images = [cv2.imread(os.path.join(src, name), -1) for name in os.listdir(src)]
    images = [img / max_value for img in images]

    args.reference = args.reference if args.reference >= 0 else len(images) // 2

    ax = plt.axes()

    plt.xlabel('Relative Camera Exposure (1/3 stops)')
    plt.ylabel('Relative Pixel Values (log2)')
    if args.title:
        plt.title(args.title)

    sample_coords = [(x, y) for x in range(1, images[0].shape[0] - 1) for y in range(1, images[0].shape[1] - 1)]

    if args.like:
        selector = np.array([float(x) for x in args.like.split(',')])
        selector_inv = 1 - selector
        def score(coord):
            x, y = coord
            color = images[args.reference][x, y]
            score = np.sum(color * selector - color * selector_inv) / np.sum(color)
            return score
        sample_coords.sort(key=score, reverse=True)
        sample_coords = sample_coords[:args.count]
        print('Samples:')
        for x, y in sample_coords:
            print ('\t%d, %d' % (x, y))

    if args.sample:
        sample_coords = [[int(x) for x in s.split(',')] for s in args.sample]


    indices = range(len(images))

    for x, y in sample_coords:

        if args.all_channels:
            for c in (0, 1, 2):
                values = [np.log2(images[i][x, y, c]) for i in indices]
                color = [int(i == c) for i in (0, 1, 2)]
                ax.plot(indices, values, '-', color=color)

        else:
                values = [np.log2(np.sum(images[i][x, y]) / 3) for i in indices]
                color = images[args.reference][x, y]
                ax.plot(indices, values, '-', color=color)



    plt.xlim(0, len(images) - 1)


    if args.dst:
        plt.savefig(args.dst, bbox_inches=0)
    else:
        plt.show()


if __name__ == '__main__':

    try:
        src = "C:/Hoa_Python_Projects/hdr_opencv/pictures_forhdr/normal/1/"
        plot_crc(src)

    except OSError  as e:
        print("Error: " + str(e))