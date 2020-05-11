import mxnet as mx

import os

import cv2
import numpy as np
from mxnet import nd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data import Dataset


def batchify_fn(data):
    """Collate data into batch."""
    if isinstance(data[0], nd.NDArray):
        return nd.stack(*data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        return nd.array(data, dtype=data.dtype)


class FaceDataset(Dataset):
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

        image_ar = np.load(ar_path)
        image_ar = cv2.resize(image_ar, (54, 54))
        image_ar = np.expand_dims(image_ar, 0)

        image = nd.array(image)
        image_ar = nd.array(image_ar)

        return image, image_ar


if __name__ == '__main__':
    batchsize = 128
    ctx = [mx.gpu(i) for i in range(4)]
    sym_ar, arg_ar, aux_ar = mx.model.load_checkpoint("model_ar", 0)
    model_ar = mx.mod.Module(sym_ar, data_names=('data_ar',), context=ctx)
    model_ar.bind(data_shapes=[('data_ar', (batchsize, 1, 54, 54))], for_training=False)
    model_ar.init_params(arg_params=arg_ar, aux_params=aux_ar)

    sym_rgb, arg_rgb, aux_rgb = mx.model.load_checkpoint("model_rgb", 0)
    model_rgb = mx.mod.Module(sym_rgb, data_names=('data_rgb', ), context=ctx)
    model_rgb.bind(data_shapes=[('data_rgb', (batchsize, 3, 54, 54))], for_training=False)
    model_rgb.init_params(arg_params=arg_rgb, aux_params=aux_rgb)

    image_list = os.listdir("/root/huzechen_pairs")
    image_list = [x for x in image_list if 'jpg' in x]
    image_list = [os.path.join("/root/huzechen_pairs", x) for x in image_list]
    val_loader = DataLoader(
        dataset=FaceDataset(image_list[:3000]),
        batch_size=batchsize,
        shuffle=False,
        sampler=None,
        last_batch='discard',
        batch_sampler=None,
        num_workers=16,
        thread_pool=False,
        prefetch=4
    )

    rgb_feat = []
    ar_feat = []
    for val_batch in val_loader:
        model_rgb.forward(mx.io.DataBatch([val_batch[0]]), is_train=False)
        model_ar.forward(mx.io.DataBatch([val_batch[1]]), is_train=False)
        rgb_feat.append(model_rgb.get_outputs(merge_multi_context=True)[0])
        ar_feat.append(model_ar.get_outputs(merge_multi_context=True)[0])
    rgb_feat = nd.concat(*rgb_feat, dim=0)
    ar_feat = nd.concat(*ar_feat, dim=0)

    index = np.arange(len(rgb_feat))
    score = []
    for idx in range(ar_feat.shape[0]):
        pos_index = np.array([idx])
        negative_class_pool = np.setdiff1d(index, pos_index)
        neg_index = np.random.choice(negative_class_pool, 10, replace=False)
        all_index = np.concatenate((pos_index, neg_index), axis=0)
        _query_ar_feat = ar_feat[idx]
        _rgb_feat = rgb_feat[all_index]
        if nd.argmax(nd.dot(_query_ar_feat, _rgb_feat, transpose_b=True), axis=0).asscalar() == 0:
            score.append(True)
        else:
            score.append(False)
    acc = sum(score) / len(score)
    print("acc:%f" % acc)