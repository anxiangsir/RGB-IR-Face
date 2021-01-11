import json
import sys
import numpy as np
import cv2
import os

def func(folder, w):
    for name in os.listdir(folder):
        list_path = list()
        list_size = list()
        img_dir = os.path.join(folder, name)
        for path in os.listdir(img_dir):
            path_abs = os.path.join(img_dir, path)
            list_path.append(path_abs)
            img = cv2.imread(path_abs)
            list_size.append(img.shape[0] * img.shape[1])
            #
        sort_idx = np.argsort(list_size)

        if len(sort_idx) < 3:
            continue
        else:
            w.write("%s,%s\n" % (list_path[sort_idx[0]], list_path[sort_idx[-1]]))


if __name__ == '__main__':
    folder = sys.argv[1]
    w = open(sys.argv[2], "w")
    func(folder, w)
