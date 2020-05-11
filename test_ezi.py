import os

import cv2
import numpy as np
import pyezi as ezi


def init_model(model_name, ezm_path):
    model = ezi.CVModel(model_name)
    model.LoadModelFromFile(ezm_path)
    return model


def rgb_preprocess(image):
    image = image.astype(np.float32)
    image = cv2.resize(image, (54, 54))
    image = image / 255
    image = np.transpose(image, (2, 0, 1))
    return image


def ir_preprocess(image):
    image = image.astype(np.float32)
    image = cv2.resize(image, (54, 54))
    image = np.expand_dims(image, -1)
    image = np.transpose(image, (2, 0, 1))
    return image


class FeatureModelInfer:
    def __init__(self, model_name, ezm_path, out_layer_name='fc1'):
        self.model = init_model(model_name, ezm_path)
        self.out_layer_name = out_layer_name

    def get_infer_result(self, image):
        self.model.UploadInput('data', image)
        self.model.Infer()
        feature = self.model.DownloadOutput(self.out_layer_name).flatten()
        feature = feature / np.linalg.norm(feature, ord=2)
        feature = np.expand_dims(feature, 0)
        return feature


if __name__ == '__main__':
    image_list = os.listdir("test")
    image_list = [x for x in image_list if 'jpg' in x]
    image_list = [os.path.join("test", x) for x in image_list]

    model_rgb = FeatureModelInfer('ModelRGB', 'ModelRGB_ncnn.ezm')
    model_ir = FeatureModelInfer('ModelIR', 'ModelIR_ncnn.ezm')
    rgb_feat = []
    ir_feat = []
    for path in image_list:
        #
        image_rgb = cv2.imread(path)
        image_rgb = rgb_preprocess(image_rgb)
        rgb_feat.append(model_rgb.get_infer_result(image_rgb))
        #
        ir_path = path.split('.')[0] + '.npy'
        image_ir = np.load(ir_path)
        image_ir = ir_preprocess(image_ir)
        ir_feat.append(model_ir.get_infer_result(image_ir))

    rgb_feat = np.concatenate(rgb_feat, axis=0)
    ir_feat = np.concatenate(ir_feat, axis=0)

    index = np.arange(len(rgb_feat))
    score = []
    for idx in range(ir_feat.shape[0]):
        pos_index = np.array([idx])
        negative_class_pool = np.setdiff1d(index, pos_index)
        neg_index = np.random.choice(negative_class_pool, 10, replace=False)
        all_index = np.concatenate((pos_index, neg_index), axis=0)
        _query_ar_feat = ir_feat[idx]
        _rgb_feat = rgb_feat[all_index]
        if np.argmax(np.dot(_query_ar_feat, _rgb_feat.T), axis=0) == 0:
            score.append(True)
        else:
            score.append(False)
    acc = sum(score) / len(score)
    print("acc:%f" % acc)

import mxnet as mx

mx.nd.multi_lars()
mx.nd.multi_sum_sq()
mx.nd.preloaded_multi_mp_sgd_mom_update()