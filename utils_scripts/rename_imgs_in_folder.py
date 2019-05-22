""" This script receives a folder and rename all image files in this folder.
    It may be useful to exec this file on a database if a ordering is
    desired to compare results.
"""

import numpy as np
import glob
from pathlib import Path
from PIL import Image
import re
import time
import argparse


def get_img_real_ext(pathname):
    """ Function that identifies the format of the image and return it as
        a lower case string. If it's not a valid image, returns None.
    """
    valid, img = is_pillow_valid_img(pathname)
    if not valid:
        return None

    format = img.format.lower()
    img.close()
    return format


def is_pillow_valid_img(pathname, return_ref=True):
    """ Function that verifies if the file is a valid image considering
        the pillow library, that's used in this code. If desired, it
        returns the opened ref. The retuned reference must be closed later.
    """
    try:
        img = Image.open(pathname)
        is_valid = True
    except Exception:
        img = []
        is_valid = False

    ret = is_valid
    if return_ref:
        ret = list([ret])
        ret.append(img)

    return ret


def get_all_imgs_in_folder(folder):
    """ Function that returns the list of names of valid image files in
        a folder.
    """
    all_files = np.array(sorted(glob.glob(folder + '/*')))
    pillow_imgs = list(map(is_pillow_valid_img, all_files))
    img_files = all_files[pillow_imgs]
    img_files = np.array(list(map(lambda s: Path(s), img_files)))
    return img_files


def temp_rename_images(img_pathnames):
    """ To avoid problems when renaming, the images are given a unique
        temporary name.
    """
    imgs_len = len(img_pathnames)
    aux_names = list(map(lambda num: str(int(time.time())) + '_' + str(num),
                         np.arange(imgs_len)))
    list(map(lambda path, str: path.rename(path.parent / str), img_pathnames,
             aux_names))


def final_rename_images(img_pathnames, ext_array):
    """ The final image names in ordered numeric names with extensions in
        agreement with the real image standard
    """
    # Format numbering to put the strings in ascending order
    imgs_len = len(img_pathnames)
    num_digits = len(str(imgs_len))
    digits_format = '{:0' + str(num_digits) + '.0f}'
    img_pattern = '{}.{}'.format(digits_format, '{}')
    new_names = list(map(lambda num, ext: img_pattern.format(num, ext),
                         np.arange(imgs_len), ext_array))
    list(map(lambda path, s: path.rename(path.parent / s), img_pathnames,
             new_names))


def standardize_images_names(folder, img_regex):
    """ Function that analyzes and prepare the folder to be used by the
        generator
    """
    img_pathnames = get_all_imgs_in_folder(folder)
    imgs_len = len(img_pathnames)
    ext_list = list(map(get_img_real_ext, img_pathnames))

    # Match the name with the image format as suffix in the name
    formatted_index = list(map(lambda path, format: re.fullmatch(
        img_regex.format(format), path.name), img_pathnames, ext_list))
    formatted_imgs_len = len(np.where(formatted_index)[0])
    if imgs_len != formatted_imgs_len and imgs_len > 0:
        # Use a unique temp name for images
        temp_rename_images(img_pathnames)
        # Get all images again
        img_pathnames = get_all_imgs_in_folder(folder)
        # Definitive renaming
        final_rename_images(img_pathnames, ext_list)


if __name__ == '__main__':
    # Regex to verify the formatted image names
    img_regex = '([0-9]+)\.{}'

    parser = argparse.ArgumentParser(description='Rename img folders for use.')
    parser.add_argument('--path', action='store', type=str, required=True,
                        help='Folder containing the images')
    args = parser.parse_args()
    standardize_images_names(args.path, img_regex)