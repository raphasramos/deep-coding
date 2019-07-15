""" File containing the implementation of data loaders from images """

from .processing import ImgProc
from .enums import ImgData

import glob
import numpy as np
from pathlib import Path
from itertools import repeat, cycle
from multiprocessing.pool import ThreadPool
from multiprocessing import Process as MultiProcess, Queue, Event as MultiEvent
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock, Event as ThreadEvent
from threading import Condition
from collections import deque
from setproctitle import setproctitle
import pandas as pd
import os


# Thread shared variable for variables loaded
loaded_list = deque()
data_lock = Lock()
get_condition, load_condition = Condition(data_lock), Condition(data_lock)
thread_stop_event = ThreadEvent()
max_len = 100


class Generator:
    """ Class representing an object that analyses an image database and
    generate data from it """

    def __init__(self, input_shape, generator_conf, queue_size):
        if not Path(generator_conf['path']).exists():
            raise FileNotFoundError(generator_conf['path'] + ' doesn\'t exist!')

        self.gen_conf = generator_conf.copy()
        self._input_shape = input_shape
        self._queue_size = queue_size
        self._gen_stop_event = None
        self._queue = None
        self._proc = None
        self._do_img_indexing()
        self._iterations = 0
        self._img_paths = []

    def get_db_files_pathnames(self):
        """ Method that returns the pathname of the files in the generator
            folders. It can contain non valid images. So you must test before
            opening them.
        """
        return self._img_paths

    def get_patch_size(self):
        """ Get the size of the patches """
        return self._input_shape[2]

    def get_db_folder(self):
        """ Get the folder of the database used """
        return Path(self.gen_conf['path'])

    def _create_index_file(self, index_pathname):
        """ Function that creates the index file of a folder """
        print('Indexing file does not exist! Indexing database!\n')
        pool = ThreadPool()
        all_files = np.array(
            sorted(glob.glob(self.gen_conf['path'] + '/**/*', recursive=True)))
        valid = pool.starmap(ImgProc.is_pillow_valid_img,
                             zip(all_files, repeat(False)))
        img_paths = np.array(all_files, dtype=object)[valid]
        w, h = list(zip(*pool.starmap(ImgProc.get_size, zip(img_paths))))
        img_paths = list(map(os.path.relpath, all_files[valid],
                             repeat(self.gen_conf['path'])))
        df = pd.DataFrame(zip(w, h), index=pd.Index(img_paths, name='paths'),
                          columns=pd.Index(['width', 'height']))
        df.to_csv(index_pathname)
        pool.close(), pool.join()

    def _do_img_indexing(self):
        """ Function that analyses and writes a file describing all images
            in a folder with its size. If the file already exists, it reads
            and index img files in the folders
        """
        index_pathname = Path(self.gen_conf['path']) / 'index.csv'
        if not index_pathname.exists():
            self._create_index_file(index_pathname)
        data = pd.read_csv(index_pathname)
        paths, width, height = list(zip(*data.to_numpy()))
        self._img_paths = list(map(
            lambda p: Path(self.gen_conf['path']) / p, paths))
        self._img_paths = np.array(self._img_paths, dtype=object)

        patches = list(map(ImgProc.calc_n_patches, zip(width, height),
                           repeat(self._input_shape[1])))
        num_patches = np.sum(patches)
        iterations = np.ceil(num_patches / self._input_shape[0]).astype(int)
        self._iterations = iterations

    def get_iter(self):
        """ Get the number of iterations for this generator. This is true
            only in generators of training databases
        """
        return self._iterations

    @staticmethod
    def _load_thread_task(files_list):
        """ Auxiliary task to load data asynchronously """
        global loaded_list, thread_stop_event, max_len, get_condition
        args = cycle(np.array_split(files_list,
                                    np.arange(20, len(files_list), 20)))
        del files_list
        load_pool = ThreadPool()
        for arg in args:
            load_condition.acquire()
            while len(loaded_list) >= max_len:
                load_condition.wait(timeout=1.2)
            load_condition.release()
            data = load_pool.starmap(ImgProc.load_image,
                                     zip(arg, repeat(ImgData.FLOAT)))
            data_lock.acquire()
            loaded_list.append(data)
            get_condition.notify_all()
            data_lock.release()
            if thread_stop_event.is_set():
                load_pool.close()
                load_pool.join()
                break

    @staticmethod
    def _get_next_chunk():
        """ Function that gets the data shared between the threads """
        global load_condition, loaded_list, max_len, thread_stop_event
        get_condition.acquire()
        while len(loaded_list) == 0:
            get_condition.wait(timeout=0.4)
        imgs = loaded_list.popleft()
        if len(loaded_list) < max_len:
            load_condition.notify_all()
        get_condition.release()
        return imgs

    def _data_task(self):
        """ The task of the database generator responsible for collecting
            data.
        """
        global data_lock, loaded_list, thread_stop_event, get_condition
        setproctitle('python3 - _data_task')
        # Evaluation mode
        thread = Thread(target=self._load_thread_task, args=(self._img_paths,))
        thread.start()
        iteration, patches = 0, []
        while not self._gen_stop_event.is_set():
            imgs = self._get_next_chunk()
            patches += [np.vstack(list(map(ImgProc.extract_img_patch, imgs,
                                           repeat(self._input_shape[1]))))]
            num_patches = np.sum(list(map(lambda e: len(e), patches)))
            if num_patches < self._input_shape[0]:
                continue
            patches = np.vstack(patches)
            indexes = np.arange(self._input_shape[0], len(patches) + 1,
                                self._input_shape[0])
            batches = list(filter(len, np.array_split(patches, indexes)))
            patches = [batches.pop(-1)] if len(batches) > 1 else []
            iteration += len(batches)
            if iteration > self._iterations:
                self._gen_stop_event.set()
                diff = iteration - self._iterations
                batches = batches[:-diff]
            list(map(self._queue.put, batches))

        thread_stop_event.set()
        if len(patches):
            patches = patches[0]
            pad_width = self._input_shape[0] - len(patches)
            pad_shape = [(0, pad_width)] + (len(patches.shape) - 1) * [(0, 0)]
            patches = np.pad(patches, pad_shape, 'symmetric')
            self._queue.put(patches)

        thread.join()
        data_lock.acquire()
        loaded_list.clear()
        data_lock.release()

    def start(self):
        """ Procedures to start and wait for the generator """
        self._queue = Queue(self._queue_size)
        self._gen_stop_event = MultiEvent()
        self._proc = MultiProcess(target=self._data_task)
        self._proc.start()

    def stop(self):
        """ Stop the generator """

        if not self._proc:
            return

        self._gen_stop_event.set()
        # TODO: check this.
        #  Weirdly, the process has finished but the join hangs forever
        self._proc.join(5)
        self._proc.terminate()
        self._queue.close()
        self._proc = None

    def get_batch(self):
        """ Get a batch from the queue """
        batch = self._queue.get()
        return batch

