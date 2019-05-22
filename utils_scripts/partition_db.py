""" Script that receives a folder, maps all files in that folder and partition
    this dataset into training and testing parts.
"""

import numpy as np
import glob
from pathlib import Path
from shutil import copy, rmtree
from os.path import isfile
from multiprocessing.pool import ThreadPool

# Adjustable parameters
sub_size = 100000
in_db_folder = '../database0'
train_percent = 0.9
out_db_folder = Path('eduardo_partitioned/database0')

if __name__ == '__main__':
    if out_db_folder.exists():
        rmtree(out_db_folder, ignore_errors=True)
    out_db_folder.mkdir(parents=True, exist_ok=True)

    all_imgs = glob.glob(in_db_folder + '/**/*', recursive=True)
    all_imgs = np.array(all_imgs)[list(map(lambda p: isfile(p), all_imgs))]
    num_imgs = len(all_imgs)
    np.random.shuffle(all_imgs)
    all_imgs = list(map(lambda p: Path(p), all_imgs))
    partitioned_data = np.array_split(
        all_imgs, [round(train_percent * len(all_imgs))])
    new_folders = [out_db_folder / 'train', out_db_folder / 'test']
    for part_data, folder in zip(partitioned_data, new_folders):
        sub_partitions = np.array_split(
            part_data, np.arange(sub_size, len(part_data), sub_size))
        sub_folder = list(map(lambda n: folder / Path(str(n)),
                              np.arange(len(sub_partitions))))
        list(map(lambda f: f.mkdir(parents=True, exist_ok=True), sub_folder))
        full_names = []
        for sub_p, sub_f in zip(sub_partitions, sub_folder):
            full_names.append(list(map(
                lambda n, p: sub_f / (sub_f.name + '_' + str(n)
                                      + str(p.suffix)),
                np.arange(len(sub_p)), sub_p)))
        pool = ThreadPool()
        for new_paths, old_paths in zip(full_names, sub_partitions):
            pool.starmap(copy, zip(old_paths, new_paths))

