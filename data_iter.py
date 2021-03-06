import os
import shutil

import cv2
import numpy as np
from mxnet import nd
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms
import mxnet as mx


def batchify_fn(data):
    return mx.nd.concat(*data, dim=0)


class FaceDataset(Dataset):
    def __init__(self, path_list):
        assert isinstance(path_list, list)
        self.path_list = path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        #
        path_small, path_large = self.path_list[idx].split(",")
        img_small = cv2.imread(path_small).astype(np.float32)
        img_large = cv2.imread(path_large).astype(np.float32)

        img_small = cv2.resize(img_small, (128, 128))
        img_large = cv2.resize(img_large, (128, 128))

        img_small = nd.expand_dims(nd.transpose(mx.nd.array(img_small), (2, 0, 1)), axis=0)
        img_large = nd.expand_dims(nd.transpose(mx.nd.array(img_large), (2, 0, 1)), axis=0)

        return mx.nd.concat(img_small, img_large, dim=0)


        # image_rgb = cv2.imread(self.path_list[idx])
        # image_rgb = image_rgb.astype(np.float32)
        # image_rgb = cv2.resize(image_rgb, (54, 54))
        # image_rgb = image_rgb / 255
        # if nd.random.randint(0, 100) > 50:
        #     image_rgb = np.pad(image_rgb, pad_width=((5, 5), (5, 5), (0, 0)),
        #                        constant_values=0, mode='constant')
        #     image_rgb = nd.array(image_rgb)
        #     image_rgb = transforms.image.random_crop(image_rgb, (54, 54))[0]
        # else:
        #     image_rgb = nd.array(image_rgb)
        # image_rgb = nd.transpose(image_rgb, (2, 0, 1))
        #
        # image_ir = np.load(ar_path)
        # image_ir = cv2.resize(image_ir, (54, 54))
        # image_ir = np.expand_dims(image_ir, -1)
        #
        # # random jitter
        # if nd.random.randint(0, 100) < 10:
        #     tmp_jitter = np.random.randint(-5, 5)
        #     image_ir += tmp_jitter * 1.
        # # random scale
        # if nd.random.randint(0, 100) < 10:
        #     scale_jitter = np.random.randint(90, 110) / 100
        #     image_ir *= scale_jitter
        # # random crop
        # if nd.random.randint(0, 100) > 50:
        #     image_ir = np.pad(image_ir, pad_width=((5, 5), (5, 5), (0, 0)),
        #                       constant_values=0, mode='constant')
        #     image_ir = nd.array(image_ir)
        #     image_ir = transforms.image.random_crop(image_ir, (54, 54))[0]
        # else:
        # #     image_ir = nd.array(image_ir)
        # image_ir = nd.transpose(image_ir, (2, 0, 1))
        # return image_rgb, image_ir


class FaceDatasetTest(Dataset):
    def __init__(self, path_list):
        assert isinstance(path_list, list)
        self.path_list = path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        #
        rgb_path = self.path_list[idx]
        ar_path = rgb_path.split('.')[0] + '.npy'

        image = cv2.imread(self.path_list[idx])
        image = image.astype(np.float32)
        image = cv2.resize(image, (54, 54))
        image = image / 255
        image = np.transpose(image, (2, 0, 1))

        image_ir = np.load(ar_path)
        image_ir = cv2.resize(image_ir, (54, 54))
        image_ir = np.expand_dims(image_ir, 0)

        image = nd.array(image)
        image_ir = nd.array(image_ir)

        return image, image_ir


if __name__ == '__main__':
    image_list = os.listdir("/root/huzechen_pairs")
    image_list = [x for x in image_list if 'jpg' in x][:1000]
    for i in image_list:
        shutil.copy(
            os.path.join("/root/rgb2ar", i),
            os.path.join("/root/test", i)
        )

        shutil.copy(
            os.path.join("/root/rgb2ar", i.split('.')[0] + '.npy'),
            os.path.join("/root/test", i.split('.')[0] + '.npy')
        )
