import os

import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon.data import DataLoader
from mxnet.lr_scheduler import PolyScheduler

from data_iter import FaceDataset
from logger import setlogger
import mxnet.autograd as ag
from resnet import get_symbol

#
ctx = [mx.gpu(i) for i in range(4)]
batch_size = 128
max_update = 10000
base_lr = 0.01
model_root = "./"
#
image_list = os.listdir("/root/huzechen_pairs")
image_list = [x for x in image_list if 'jpg' in x]
image_list = [os.path.join("/root/huzechen_pairs", x) for x in image_list]


def create_symbol():
    data_rgb = mx.sym.var("data_rgb")
    data_ir = mx.sym.var("data_ir")
    sym_rgb = get_symbol(
        data=data_rgb, num_classes=128, num_layers=18, version_output='H',
        version_input=1, version_se=0, version_unit=1, version_act='relu',
        dtype='float32', memonger=False, use_global_stats=False)
    sym_ir = get_symbol(
        data=data_ir, num_classes=128, num_layers=18, version_output='H',
        version_input=1, version_se=0, version_unit=1, version_act='relu',
        dtype='float32', memonger=False, use_global_stats=False)
    return sym_rgb, sym_ir


class ContrastiveLoss(object):
    def __init__(self, l2_norm=True, temperature=0.07):
        self.l2_norm = l2_norm
        self.t = temperature
        pass

    @staticmethod
    def norm_func(x1, x2):
        return nd.L2Normalization(x1), nd.L2Normalization(x2)

    def __repr__(self):
        return "ContrastiveLoss: t: [%.4f]" % self.t

    def __call__(self, query, key):
        if self.l2_norm:
            query, key = self.norm_func(query, key)

        sim = nd.dot(query, key, transpose_a=False, transpose_b=True) / self.t
        sim_exp = nd.exp(sim)
        denominator = nd.sum(sim_exp, axis=1)
        numerator = nd.diag(sim_exp)
        objective = - nd.log(numerator / denominator)
        return objective


def main():
    data_loader = DataLoader(
        dataset=FaceDataset(image_list[1000:]),
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        last_batch='discard',
        batch_sampler=None,
        num_workers=16,
        thread_pool=False,
        prefetch=4
    )

    val_loader = DataLoader(
        dataset=FaceDataset(image_list[:1000]),
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        last_batch='discard',
        batch_sampler=None,
        num_workers=16,
        thread_pool=False,
        prefetch=4
    )

    step = 0
    logger = setlogger(models_root=model_root, rank=0)
    sym_rgb, sym_ir = create_symbol()
    mod_rgb = mx.mod.Module(sym_rgb, ('data_rgb',), context=ctx)
    mod_ir = mx.mod.Module(sym_ir, ('data_ir',), context=ctx)
    #
    mod_rgb.bind([('data_rgb', (batch_size, 3, 54, 54))])
    mod_ir.bind([('data_ir', (batch_size, 1, 54, 54))])
    #
    mod_rgb.init_params()
    mod_ir.init_params()
    #
    mod_rgb.init_optimizer(
        optimizer='sgd', optimizer_params={
            'learning_rate': base_lr,
            'lr_scheduler': PolyScheduler(max_update, base_lr),
            'momentum': 0.9,
            'wd': 5e-4,
            'rescale_grad': 1 / batch_size,
        })

    mod_ir.init_optimizer(
        optimizer='sgd', optimizer_params={
            'learning_rate': base_lr,
            'lr_scheduler': PolyScheduler(max_update, base_lr),
            'momentum': 0.9,
            'wd': 5e-4,
            'rescale_grad': 1 / batch_size,
        })

    while True:
        for batch in data_loader:
            mx.nd.waitall()
            step += 1
            mod_rgb.forward(mx.io.DataBatch([batch[0]]), is_train=True)
            mod_ir.forward(mx.io.DataBatch([batch[1]]), is_train=True)

            feat_rgb = mod_rgb.get_outputs(merge_multi_context=True)[0]
            feat_ir = mod_ir.get_outputs(merge_multi_context=True)[0]

            feat_rgb.attach_grad()
            feat_ir.attach_grad()
            c = ContrastiveLoss()
            with ag.record():
                l2loss = c(feat_rgb, feat_ir) + c(feat_ir, feat_rgb)
            l2loss.backward()
            logger.info("step:%d loss:%f" % (step, nd.sum(l2loss).asscalar()))
            mod_rgb.backward(out_grads=[feat_rgb.grad])
            mod_ir.backward(out_grads=[feat_ir.grad])
            mod_rgb.update()
            mod_ir.update()

            if step % 1000 == 0:
                mod_rgb.save_checkpoint(os.path.join(model_root, "model_rgb"), 0)
                mod_ir.save_checkpoint(os.path.join(model_root, "model_ir"), 0)

            if step % 300 == 0:
                rgb_feat = []
                ir_feat = []
                for val_batch in val_loader:
                    mod_rgb.forward(mx.io.DataBatch([val_batch[0]]), is_train=False)
                    mod_ir.forward(mx.io.DataBatch([val_batch[1]]), is_train=False)
                    rgb_feat.append(mod_rgb.get_outputs(merge_multi_context=True)[0])
                    ir_feat.append(mod_ir.get_outputs(merge_multi_context=True)[0])
                rgb_feat = nd.concat(*rgb_feat, dim=0)
                ir_feat = nd.concat(*ir_feat, dim=0)

                index = np.arange(len(rgb_feat))
                score = []
                for idx in range(ir_feat.shape[0]):
                    pos_index = np.array([idx])
                    negative_class_pool = np.setdiff1d(index, pos_index)
                    neg_index = np.random.choice(negative_class_pool, 10, replace=False)
                    all_index = np.concatenate((pos_index, neg_index), axis=0)
                    _query_ir_feat = ir_feat[idx]
                    _rgb_feat = rgb_feat[all_index]
                    if nd.argmax(nd.dot(_query_ir_feat, _rgb_feat, transpose_b=True), axis=0).asscalar() == 0:
                        score.append(True)
                    else:
                        score.append(False)
                acc = sum(score) / len(score)
                logger.info("step:%d acc:%f" % (step, acc))


if __name__ == '__main__':
    main()
