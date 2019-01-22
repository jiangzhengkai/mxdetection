import numpy as np
import math
import mxnet as mx
import cv2
import logging
import multiprocessing
from mxnet.executor_manager import _split_input_slice

class BaseIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size, ctx=None, work_load_list=None):
        self.roidb = roidb
        self.config = config
        self.batch_size = batch_size
        self.data = None
        self.label = None
        
        self.shuffle = config.TRAIN.aug_strategy.shuffle
        self.aspect_grouping = config.TRAIN.aug_strategy.aspect_grouping
        
        self.ctx = [mx.cpu()] if ctx is None else ctx
        if work_load_list is None:
            work_load_list = [1] * len(self.ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(self.ctx), 'Invalid settings for work load.'
        self.work_load_list = work_load_list
        
        self.has_load_data = False
        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.reset()
    
    def load_data(self):
        assert not self.has_load_data
        self.has_load_data = True
        self.imgrec = None
        num_imgrec = len(self.config.dataset.train_imgrec_path_list)
        if num_imgrec > 0:
            assert num_imgrec == len(self.config.dataset.train_imgidx_path_list)
            self.imgrec = []
            for i in range(num_imgrec):
                imgidx_path = self.config.dataset.train_imgidx_path_list[i]
                imgrec_path = self.config.dataset.train_imgrec_path_list[i]
                self.imgrec.append(mx.recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r'))
            logging.info('use imgrec for training')
        
    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                lim = math.floor(len(horz_inds) / self.batch_size) * self.batch_size
                horz_inds = np.random.choice(horz_inds, size=int(lim), replace=False) if lim != 0 else []
                lim = math.floor(len(vert_inds) / self.batch_size) * self.batch_size
                vert_inds = np.random.choice(vert_inds, size=int(lim), replace=False) if lim != 0 else []
                inds = np.hstack((horz_inds, vert_inds))
                inds_ = np.reshape(inds, (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds.astype(np.int32)
                self.size = len(self.index)
            else:
                np.random.shuffle(self.index)
    
    @property
    def provide_data(self):
        if self.data is None:
            self.get_batch()
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        if self.label is None:
            self.get_batch()
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def iter_next(self):
        return self.cur + self.batch_size <= self.size
        
    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data,
                                   label=self.label,
                                   pad=0,
                                   provide_data=self.provide_data,
                                   provide_label=self.provide_label)
        else:
            raise StopIteration
        
    def get_batch(self):
        if not self.has_load_data:
            self.load_data()
        index_start = self.cur
        index_end = self.cur + self.batch_size
        roidb = [self.roidb[self.index[i]] for i in range(index_start, index_end)]
        slices = _split_input_slice(index_end - index_start, self.work_load_list)

        data_list = []
        label_list = []
        for i_slice in slices:
            i_roidb = [roidb[i] for i in range(i_slice.start, i_slice.stop)]
            for j in range(len(i_roidb)):
                data, label = self.get_one_roidb(i_roidb[j], j)
                data_list.append(data)
                label_list.append(label)

        all_data = dict()
        for name in self.data_name:
            all_data[name] = tensor_vstack([data[name] for data in data_list])

        all_label = dict()
        for name in self.label_name:
            pad = -1 if 'label' in name and 'weight' not in name else 0
            all_label[name] = tensor_vstack([label[name] for label in label_list], pad=pad)

        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]

    def get_one_roidb(self, roidb_j, j=0):
        return [], []
        
class BaseTestIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size):
        self.roidb = roidb
        self.config = config
        self.batch_size = batch_size
        self.data = None
        
        self.has_load_data = False
        
        self.cur = 0
        self.size = len(self.roidb)
        self.index = np.arange(self.size)
        self.reset()
        
        self.data_batch = []

    def load_data(self):
        assert not self.has_load_data
        self.has_load_data = True
        self.imgrec = None
        num_imgrec = len(self.config.dataset.test_imgrec_path_list)
        if num_imgrec > 0:
            assert num_imgrec == len(self.config.dataset.test_imgidx_path_list)
            self.imgrec = []
            for i in range(num_imgrec):
                imgidx_path = self.config.dataset.test_imgidx_path_list[i]
                imgrec_path = self.config.dataset.test_imgrec_path_list[i]
                self.imgrec.append(mx.recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r'))
            logging.info('use imgrec for test')
    
    def reset(self):
        self.cur = 0

    @property
    def provide_data(self):
        if self.data is None:
            self.get_batch()
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data_batch
        else:
            raise StopIteration
        
    def get_batch(self):
        if not self.has_load_data:
            self.load_data()
        self.clear_local_vars()
        index_start = self.cur
        index_end = min(index_start + self.batch_size, self.size)
        for i in range(index_start, index_end):
            data, need_forward = self.get_one_roidb(self.roidb[self.index[i]])
            self.data = [mx.nd.array(data[name]) for name in self.data_name]
            self.data_batch.append(mx.io.DataBatch(data=self.data, label=[], pad=0, provide_data=self.provide_data))
            self.need_forward.append(need_forward)
    def get_one_roidb(self, roidb_j, j=0):
        return []
    def clear_local_vars(self):
        self.data_batch = []
        self.need_forward = []
        self.extra_local_vars = []
def worker_loop(data_iter, key_queue, data_queue, shut_down):
    key_queue.cancel_join_thread()
    data_queue.cancel_join_thread()
    while True:
        if shut_down.is_set():
            break
        batch_str = key_queue.get()
        if batch_str is None:
            break
        data_iter.index = [int(batch_id) for batch_id in batch_str.split()]
        assert len(data_iter.index) == data_iter.batch_size
        data_iter.cur = 0
        data_queue.put(data_iter.next())
    logging.info('goodbye')       
 
class PrefetchingIter(mx.io.DataIter):
    def __init__(self, data_iter, num_workers=multiprocessing.cpu_count(), max_queue_size=8):
        super(PrefetchingIter, self).__init__()
        logging.info('num workers: %d' % num_workers)
        
        self.data_iter = data_iter
        self.size = data_iter.size
        self.batch_size = data_iter.batch_size
        self.data_name = data_iter.data_name
        self.label_name = data_iter.label_name
        self.max_data_shape = data_iter.max_data_shape
        self.max_label_shape = data_iter.max_label_shape
        
        self.num_batches = self.size / self.batch_size
        assert self.size % self.batch_size == 0
        
        self.num_workers = num_workers
        self.workers = []
        self.cur = 0
        self.key_queue = mx.gluon.data.dataloader.Queue()
        self.data_queue = mx.gluon.data.dataloader.Queue(max_queue_size)
        self.key_queue.cancel_join_thread()
        self.data_queue.cancel_join_thread()
        self.shut_down = multiprocessing.Event()
        self._create_workers()

        import atexit
        atexit.register(lambda a: a.__del__(), self)

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        return self.data_iter.provide_label

    def _create_workers(self):
        for i in range(self.num_workers):
            worker = multiprocessing.Process(target=worker_loop,
                                             args=(self.data_iter, self.key_queue, self.data_queue, self.shut_down))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _close_workers(self):
        for worker in self.workers:
            worker.join()
        self.workers = []

    def shutdown(self):
        self.shut_down.set()
        for i in range(len(self.workers)):
            self.key_queue.put(None)
        try:
            while not self.data_queue.empty():
                self.data_queue.get()
        except IOError:
            pass

    def __del__(self):
        self.shutdown()

    def reset(self):
        self.data_iter.reset()
        self.cur = 0

    def iter_next(self):
        return self.cur < self.num_batches

    def next(self):
        if self.cur == 0:
            index = self.data_iter.index.reshape((self.num_batches, self.data_iter.batch_size))
            for i in range(index.shape[0]):
                batch_str = '%d' % index[i, 0]
                for j in range(1, index.shape[1]):
                    batch_str += ' %d' % index[i, j]
                self.key_queue.put(batch_str)
        if self.iter_next():
            self.cur += 1
            return self.data_queue.get()
        else:
            raise StopIteration
        
    
    
    
