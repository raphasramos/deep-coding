""" Script that receives an image database and generates a patch version of
    this database
"""

import argparse
import numpy as np
from pathlib import Path
from os import walk
from itertools import repeat
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from PIL import Image
from skimage.util import view_as_blocks
from shutil import rmtree
from sys import exit


def _get_all_files_in_a_folder(folder):
    """ Function that returns a list of files in all subfolders. """
    all_files = []
    walk_results = list(walk(folder))
    for results in walk_results:
        full_path = list(map(lambda parent, folder: Path(parent) / folder,
                             repeat(results[0]), results[2]))
        all_files.extend(full_path)
    all_files = list(filter(None, all_files))
    all_files = np.array(all_files)
    return all_files


def read_args():
    """ Function that reads the commandline arguments """
    parser = argparse.ArgumentParser(description='Generate patches from images')
    parser.add_argument('--db_path', action='store', type=str, required=True,
                        help='Folder containing the images')
    parser.add_argument('--out_path', action='store', type=str, required=True,
                        help='Folder to store the output patches')
    parser.add_argument('--patch_size', action='store', type=str, required=True,
                        help='The size of the output patches')
    args = parser.parse_args()
    return args.db_path, args.out_path, int(args.patch_size)


def _load_img(path):
    """ Function that loads an image """
    try:
        img_ref = Image.open(path)
        img_data = np.array(img_ref)
        return img_data
    except Exception:
        return []


def _load_task(img_queue, files_list):
    """ Auxiliary task to load data asynchronously """
    pool = ThreadPool()
    img_pos = 0
    num_files = len(files_list)
    while img_pos < num_files:
        results = []
        cont = 0
        while img_pos < num_files:
            results.append(pool.apply_async(_load_img, [files_list[img_pos]]))
            img_pos += 1
            cont += 1
            if cont > 30:
                break

        for res in results:
            res = res.get()
            if len(res):
                img_queue.put(res)
    img_queue.put([])
    pool.close()
    pool.join()


def pad_img(img, patch_size, padding_method='symmetric'):
    """ Method that receives an image and a size for the patches. The method
        pad the image so that they can be cropped later
    """
    orig_shape = np.array(img.shape[:2])
    new_shape = patch_size * np.ceil(orig_shape / patch_size).astype(int)
    points_to_pad = new_shape - orig_shape
    padded_img = np.pad(img, [(0, points_to_pad[0]), (0, points_to_pad[1]),
                              (0, 0)], padding_method)
    return padded_img


def extract_img_patches(img, patch_size):
    """ Method that receives an image and the patch size and extract
        the patches of the image.
    """
    padded_img = pad_img(img, patch_size)
    color = 1
    if len(padded_img.shape) > 2:
        color = padded_img.shape[2]
    patches = view_as_blocks(padded_img, (patch_size, patch_size, color))
    patches = patches.reshape(-1, patch_size, patch_size, color)
    return patches


def save_img_from_array(data, pathname):
    """ Save an image from its data array. """
    Image.fromarray(data).save(pathname)


def _verify_out_path(out_path):
    """ Function that verifies if the output path already exists. It proposes
        the user choices in each case.
    """
    if out_path.exists():
        answer = input('The folder already exists. Whish to remove it? (y/n)')
        if answer == 'y' or answer == 'yes':
            rmtree(out_path, ignore_errors=True)
        else:
            print('Cannot overwrite folder. Choose another folder.')
            print('The script has stopped!')
            exit(1)
    out_path.mkdir(parents=True)


def process_and_save_patches(out_path, all_files, patch_size):
    """ Function that processes and saves the patches from the images """
    img_queue = Queue(150)
    load_proc = Process(target=_load_task, args=(img_queue, all_files))
    load_proc.start()
    save_pool = ThreadPool()
    cont_patches = 0

    out_path = Path(out_path)
    _verify_out_path(out_path)

    dir_format = 'dir{}'
    while True:
        img = img_queue.get()
        if not len(img):
            break
        patches = extract_img_patches(img, patch_size)
        num_patches = len(patches)
        begin = cont_patches
        end = cont_patches + num_patches
        curr_dir = out_path / Path(dir_format.format(begin // 50000))
        if not curr_dir.exists():
            curr_dir.mkdir()
        names = list(map(
            lambda n: curr_dir / ('patch_' + str(n) + '.png'),
            np.arange(begin, end)))
        save_pool.starmap_async(save_img_from_array, zip(patches, names))
        cont_patches = end
    save_pool.close()
    save_pool.join()


def main():
    in_path, out_path, patch_size = read_args()
    all_files = _get_all_files_in_a_folder(in_path)
    process_and_save_patches(out_path, all_files, patch_size)


if __name__ == '__main__':
    main()
