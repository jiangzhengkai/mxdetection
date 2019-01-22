import numpy as np
import permutohedral as Perm


class Densecrf(object):
    def __init__(self, data_shape, max_iter=10, eps=1e-20):
        self.max_iter = max_iter
        self.eps = eps
        assert len(data_shape) == 3 and data_shape[0] == 3
        self.const_ones = np.ones((data_shape[1] * data_shape[2], 1), dtype=np.float32)

    def _softmax(self, x):
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        return y

    def run_inference(self, unary, lattice_list, w_list, out_label=True):
        height = unary.shape[0]
        width = unary.shape[1]
        channels = unary.shape[2]
        unary = unary.reshape((-1, channels))
        norm_list = []
        for lattice in lattice_list:
            norm = np.zeros((unary.shape[0], 1), dtype=np.float32)
            lattice.compute(self.const_ones, out=norm)
            norm += self.eps
            norm_list.append(norm)
        res = np.zeros(unary.shape, dtype=np.float32)
        pairwise = unary.copy()
        for i in range(self.max_iter):
            softmax_pairwise = self._softmax(pairwise)
            pairwise[:] = unary
            for w, lattice, norm in zip(w_list, lattice_list, norm_list):
                lattice.compute(softmax_pairwise, out=res)
                pairwise += w * (res / norm)

        if out_label:
            label = np.squeeze(pairwise.argmax(axis=1).astype('int32'))
            label = label.reshape((height, width))
            return label
        else:
            pairwise = pairwise.reshape((height, width, channels))
            pairwise = pairwise.transpose(2, 0, 1)
            return pairwise

    @staticmethod
    def get_unary(prob, height, width):
        unary = prob.astype(dtype=np.float32)
        unary = unary[:, :height, :width]
        return unary.transpose(1, 2, 0)

    @staticmethod
    def get_gauss_lattice(height, width, gauss_sigma):
        features = np.empty((height * width, 2), dtype=np.float32)
        xv, yv = np.meshgrid(range(width), range(height))
        features[:, 0] = xv.ravel() / gauss_sigma
        features[:, 1] = yv.ravel() / gauss_sigma
        lattice = Perm.create()
        lattice.init(features)
        return lattice

    @staticmethod
    def get_bi_lattice(image, bi_alpha, bi_beta):
        height = image.shape[0]
        width = image.shape[1]
        image = image.reshape((-1, image.shape[2]))
        features = np.empty((height * width, 5), dtype=np.float32)
        xv, yv = np.meshgrid(range(width), range(height))
        features[:, 0] = xv.ravel() / bi_alpha
        features[:, 1] = yv.ravel() / bi_alpha
        features[:, 2:5] = image[:] / bi_beta
        lattice = Perm.create()
        lattice.init(features)
        return lattice


def get_lattice_w_list(image, config):
    height = image.shape[0]
    width = image.shape[1]
    lattice_list = []
    w_list = []

    lattice_list.append(Densecrf.get_gauss_lattice(height, width, config.TEST.densecrf.gauss_sigma))
    w_list.append(config.TEST.densecrf.gauss_w)
    lattice_list.append(Densecrf.get_bi_lattice(image, config.TEST.densecrf.bi_alpha, config.TEST.densecrf.bi_beta))
    w_list.append(config.TEST.densecrf.bi_w)

    return lattice_list, w_list






















