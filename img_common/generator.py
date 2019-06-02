""" File containing the implementation of data loaders from images """

from .processing import ImgProc

import glob
import numpy as np
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool
import pandas as pd
import os


class Generator:
    """ Class representing an object that analyses an image database and
    generate data from it """

    def __init__(self, input_shape, generator_conf):
        if not Path(generator_conf['path']).exists():
            raise FileNotFoundError(generator_conf['path'] + ' doesn\'t exist!')

        self.gen_conf = generator_conf.copy()
        self._input_shape = input_shape
        self._img_paths = []
        self._do_img_indexing()

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

