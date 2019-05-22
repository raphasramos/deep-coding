""" Script that makes downsampling and upsampling on images and measures the
statistics (psnr and bpp) in order to compare with the original images
"""

import numpy as np
from pathlib import Path
import argparse
from multiprocessing import Pool
import glob
from PIL import Image
from skimage.measure import compare_psnr
from skimage.transform import resize
from skimage.io import imsave
from psutil import cpu_count
from itertools import repeat
import csv
import warnings
import shutil
import sys


def is_pillow_valid_img(pathname):
    """ Function that verifies if the file is a valid image considering
        the pillow library, that's used in this code
    """
    try:
        img = Image.open(pathname)
        img.close()
        is_valid = True
    except Exception:
        is_valid = False

    return is_valid


def get_imgs_in_gen_folder(path, pool):
    """ Function that returns the list of names of valid image files in
        a folder considering the pillow library
    """
    folder = path
    all_files = np.array(sorted(glob.glob(folder + '/**/*', recursive=True)))
    pillow_imgs = pool.map(is_pillow_valid_img, all_files)
    img_files = all_files[pillow_imgs]
    img_files = np.array(list(map(lambda s: Path(s), img_files)))
    return img_files


def read_args():
    """ Function that reads command line and returns the arguments. """
    ap = argparse.ArgumentParser(description='Resize images in folder.')
    ap.add_argument('--path', '-p', type=str, required=True,
                    help='Folder containing the image')
    ap.add_argument('--factor', '-f', type=int, required=True,
                    help='Factor to resize')
    ap.add_argument('--save', '-s', type=bool,
                    help='Save resized images', default=False)
    return vars(ap.parse_args())


def resize_and_compare(img_file, factor, folder_path=None, order=1):
    """" This method returns the statistics (psnr and bpp) of a downsampled
         image by factor and upsampled to original size, wrt the
         original image. The default order is the skimage one (bi-linear)
    """
    orig_img = Image.open(img_file)
    orig_img_data = np.array(orig_img)
    new_size = orig_img.size[1]//factor, orig_img.size[0]//factor
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        img_down = resize(orig_img_data, new_size, order=order,
                          preserve_range=True)
        new_img_data = resize(img_down, orig_img.size[::-1], order=order,
                              preserve_range=True).astype(np.uint8)

    psnr = compare_psnr(new_img_data, orig_img_data)
    if folder_path:
        new_name_str = '{name}_psnr{psnr:.2f}{ext}'
        new_name = new_name_str.format(name=img_file.stem, psnr=psnr,
                                       ext=img_file.suffix)
        new_name = folder_path / new_name
        imsave(new_name, new_img_data)

    return psnr


if __name__ == '__main__':
    args = read_args()
    pool = Pool(cpu_count())
    path = Path(args['path'] + '/factor_' + str(args['factor']))
    if path.exists():
        answer = input('The output folder already exists. It\'ll be erased '
                       '(y/n):')
        if answer == 'y' or answer == 'yes':
            shutil.rmtree(str(path), ignore_errors=True)
        else:
            input('Rename or move the folder to use the script (press enter).')
            sys.exit(0)
    path.mkdir()

    img_files = get_imgs_in_gen_folder(args['path'], pool)
    args_list = [img_files, repeat(args['factor'])]
    if args['save']:
        args_list.append(repeat(path))
    args = zip(*args_list)
    measures = pool.starmap_async(resize_and_compare, args)
    names = list(map(lambda p: p.name, img_files))
    psnr = measures.get()
    with open(str(path / 'statistics.csv'), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image', 'psnr'])
        csv_out.writerows(zip(names, psnr))
        csv_out.writerow(['mean', np.mean(psnr)])

    pool.close()
    pool.join()
