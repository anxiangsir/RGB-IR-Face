import os

import mxnet as mx
import mxnet.autograd as ag
from mxnet import nd
from mxnet.gluon.data import DataLoader
from mxnet.lr_scheduler import PolyScheduler

from data_iter import FaceDataset, batchify_fn
from logger import setlogger
from resnet import get_symbol

#
ctx = [mx.gpu(i) for i in range(4)]
embedding_size = 256
batch_size = 128
max_update = 10000
base_lr = 0.01
model_root = "./"
image_size = 108
#

path_list = [x.strip() for x in open("/train_tmp/images_car_reid/train_pairs.lst", "r").readlines()]


def create_symbol():
    data = mx.sym.var("data")
    sym = get_symbol(
        data=data, num_classes=embedding_size, num_layers=50, version_output='H',
        version_input=0, version_se=0, version_unit=1, version_act='relu',
        dtype='float32', memonger=False, use_global_stats=False)
    return sym


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
        dataset=FaceDataset(path_list), batch_size=batch_size, shuffle=True, sampler=None, batchify_fn=batchify_fn,
        last_batch='discard', batch_sampler=None, num_workers=16, thread_pool=False, prefetch=4)

    step = 0
    logger = setlogger(models_root=model_root, rank=0)
    mod = mx.mod.Module(create_symbol(), context=ctx)
    mod.bind([('data', (batch_size * 2, 3, image_size, image_size))])
    mod.init_params()
    mod.init_optimizer(
        optimizer='sgd', optimizer_params={
            'learning_rate': base_lr, 'lr_scheduler': PolyScheduler(max_update, base_lr),
            'momentum': 0.9, 'wd': 5e-4, 'rescale_grad': 1 / batch_size,})

    while True:
        for batch in data_loader:
            mx.nd.waitall()
            step += 1

            mod.forward(mx.io.DataBatch([batch],), is_train=True)
            feat = mod.get_outputs(merge_multi_context=True)[0]
            feat.attach_grad()
            feat = nd.reshape(feat, (-1, embedding_size * 2))

            feat_s = feat[:, 0 * embedding_size: 1 * embedding_size]
            feat_t = feat[:, 1 * embedding_size: 2 * embedding_size]

            #
            c = ContrastiveLoss()

            with ag.record():
                l2loss = c(feat_s, feat_t)
            l2loss.backward()
            logger.info("step:%d loss:%f" % (step, nd.sum(l2loss).asscalar()))

            grad_feat = nd.reshape(nd.concat(feat_s.grad, feat_t.grad, dim=-1), (-1, embedding_size))
            mod.backward(out_grads=[grad_feat])

            if step % 1000 == 0:
                mod.save_checkpoint(os.path.join(model_root, "tmp"), 0)

            if step % 300 == 0:
                pass
                # rgb_feat = []
                # ir_feat = []
                # for val_batch in val_loader:
                #     mod_rgb.forward(mx.io.DataBatch([val_batch[0]]), is_train=False)
                #     mod_ir.forward(mx.io.DataBatch([val_batch[1]]), is_train=False)
                #     rgb_feat.append(mod_rgb.get_outputs(merge_multi_context=True)[0])
                #     ir_feat.append(mod_ir.get_outputs(merge_multi_context=True)[0])
                # rgb_feat = nd.concat(*rgb_feat, dim=0)
                # ir_feat = nd.concat(*ir_feat, dim=0)
                #
                # index = np.arange(len(rgb_feat))
                # score = []
                # for idx in range(ir_feat.shape[0]):
                #     pos_index = np.array([idx])
                #     negative_class_pool = np.setdiff1d(index, pos_index)
                #     neg_index = np.random.choice(negative_class_pool, 10, replace=False)
                #     all_index = np.concatenate((pos_index, neg_index), axis=0)
                #     _query_ir_feat = ir_feat[idx]
                #     _rgb_feat = rgb_feat[all_index]
                #     if nd.argmax(nd.dot(_query_ir_feat, _rgb_feat, transpose_b=True), axis=0).asscalar() == 0:
                #         score.append(True)
                #     else:
                #         score.append(False)
                # acc = sum(score) / len(score)
                # logger.info("step:%d acc:%f" % (step, acc))


if __name__ == '__main__':
    main()
