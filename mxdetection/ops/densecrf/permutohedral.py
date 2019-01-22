import os
import ctypes
import numpy as np

PermutohedralHandle = ctypes.c_void_p

lib_path = os.path.dirname(os.path.abspath(__file__)) + '/permutohedral_c.so'
lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)


def numpy2ctypes(array):
    return array.ctypes.data_as(ctypes.c_void_p)


class Permutohedral(object):
    def __init__(self, handle):
        assert isinstance(handle, PermutohedralHandle)
        self.handle = handle

    def __del__(self):
        lib.PermutohedralFree(self.handle)

    def init(self, features):
        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features, dtype=np.float32)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(features)))
        assert features.ndim == 2
        lib.PermutohedralInit(self.handle, numpy2ctypes(features), features.shape[1], features.shape[0])

    def compute(self, data, out=None):
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=np.float32)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(data)))
        assert data.ndim == 2
        if out is None:
            out = np.zeros(data.shape, dtype=data.dtype)
        lib.PermutohedralCompute(self.handle, numpy2ctypes(out), numpy2ctypes(data), data.shape[1])

        return out


def create():
    handle = PermutohedralHandle()
    lib.PermutohedralCreate(ctypes.byref(handle))
    return Permutohedral(handle)


if __name__ == '__main__':

    # m = 5
    # n = 4
    # data = np.ones((m, n), dtype=np.float32)
    # dst = np.ones((m, n), dtype=np.float32)
    # print dst
    # i = 2
    # j = 3
    # data[i, j] = 5
    # print lib.sumMAT(numpy2ctypes(data), ctypes.c_uint(m), ctypes.c_uint(n))
    # print lib.getMAT(numpy2ctypes(data), ctypes.c_uint(m), ctypes.c_uint(n), i, j)
    # print lib.copyMAT(numpy2ctypes(data), ctypes.c_uint(m), ctypes.c_uint(n), numpy2ctypes(dst))
    # print dst

    lattice = create()
    features = np.random.random((20, 5)).astype(dtype=np.float32)
    data = np.random.random((20, 7)).astype(dtype=np.float32)
    lattice.init(features)
    out = lattice.compute(data)

    flag = 1
