""" This file has operations for image processing """


import numpy as np
import gzip
from PIL import Image
from skimage import img_as_ubyte, img_as_float
from skimage.util import view_as_blocks
from bitstring import Bits
from pathlib import PosixPath
from pathlib import Path
import warnings

from .enums import ImgData


class ImgProc:
    """ Static class that has methods to make basic operations on images """
    @staticmethod
    def pad_img(img, patch_size, method='symmetric'):
        """ Method that receives an image and a size for the patches. The method
            pad the image so that they can be cropped later
        """
        orig_shape = np.array(img.shape[:2])
        new_shape = patch_size * np.ceil(orig_shape / patch_size).astype(int)
        points_to_pad = new_shape - orig_shape
        pad_img = np.pad(img, [(0, points_to_pad[0]), (0, points_to_pad[1]),
                         (0, 0)], method)
        return pad_img

    @staticmethod
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
            ret = [ret] + [img]
        elif is_valid:
            img.close()
        return ret

    @staticmethod
    def get_img_real_ext(pathname):
        """ Function that identifies the format of the image and return it as
            a lower case string. If it's not a valid image, returns None.
        """
        valid, img = ImgProc.is_pillow_valid_img(pathname)
        if not valid:
            return None

        format_img = img.format.lower()
        img.close()
        return format_img

    @staticmethod
    # TODO: use view_as_windows to implement extraction of overlapping patches
    def extract_img_patch(orig_img, patch_size):
        """ Method that receives an image and the patch size and extract
            the patches of the image.
        """
        if np.all(np.equal(orig_img.shape, patch_size)):
            return orig_img

        img = ImgProc.pad_img(orig_img, patch_size)
        color = 1
        if len(img.shape) > 2:
            color = img.shape[2]
        patches = view_as_blocks(img, (patch_size, patch_size, color))
        patches = patches.reshape(-1, patch_size, patch_size, color)
        return patches

    @staticmethod
    def conv_data_format(img, data_format):
        """ Method that receives a valid image array and a desired format to
            convert into.
        """
        if not isinstance(data_format, ImgData):
            raise ValueError("Format argument must be an " + ImgData.__name__)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if data_format == ImgData.UBYTE:
                out_img = img_as_ubyte(img)
                return out_img

            if data_format == ImgData.FLOAT:
                out_img = img_as_float(img)
                out_img = out_img.astype(np.float32)
                return out_img

        raise ValueError('Range parameter is not recognized!')

    @staticmethod
    def load_image(path, data_format=None, color_mode='RGB'):
        """ This method receives an image pathname and the target colorspace.
            If the path points to a non valid image, it returns empty data.
        """
        valid, img = ImgProc.is_pillow_valid_img(path)
        if not valid:
            return []

        if color_mode:
            img = img.convert(color_mode)
        img_data = np.array(img)
        img.close()

        if data_format:
            img_data = ImgProc.conv_data_format(img_data, data_format)

        return img_data

    @staticmethod
    def calc_bpp_using_gzip(img, orig_pathname, bpp_proxy, pos):
        """ Function that calculates the bpp considering the gzip. It receives
            a data_array representing an image.
        """
        try:
            # If the number is not already an integer, it's not been quantized
            # So it's not fair do estimate bpp using a round version
            if np.all(np.equal(np.mod(img, 1), 0)):
                img = np.array(img).astype(np.int)

            compressed = gzip.compress(img)
            pixels_num = np.prod(ImgProc.get_size(orig_pathname))
            # Bits has many representations. length get the len in bits
            bpp_proxy[pos] = [Bits(compressed).length / pixels_num]
        except Exception as e:
            print("calc_bpp_using_gzip: " + str(e), end='\n\n')

    @staticmethod
    def calc_metric(true_ref, test_ref, metric):
        """ Method responsible for calculating all available metrics to the
            code
        """
        if isinstance(true_ref, (Path, str)):
            true_ref = Image.open(true_ref)
        if isinstance(test_ref, (Path, str)):
            test_ref = Image.open(test_ref)
        true_ref = np.array(true_ref)
        test_ref = np.array(test_ref)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            result = metric[1](true_ref, test_ref)
            result = np.iinfo(np.uint8).max if result == float('inf') \
                else result

        return result

    @staticmethod
    def save_img(img_ref, path, mode='RGB'):
        """ Method that receives a numpy array and a path. It saves an image
            through PIL
        """
        try:
            if isinstance(img_ref, (str, PosixPath)):
                img_ref = Image.open(img_ref)
            elif isinstance(img_ref, np.ndarray):
                img_ref = Image.fromarray(img_ref)

            img_ref.save(path, mode=mode)
            img_ref.close()
        except Exception as e:
            print("save_img: " + str(e), end='\n\n')

    @staticmethod
    def get_size(path):
        """ Wrapper that receives a path of an image and returns its size """
        with Image.open(path) as img_ref:
            width, height = img_ref.size
        return width, height
