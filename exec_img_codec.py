""" This script is the entry point for the execution of the image
    autoencoder. It reads the img_codec_conf.json
"""
import argparse
from shutil import copyfile
import json
from pathlib import Path
import numpy as np
import platform
import os

from img_common.autoencoder import AutoEnc


np.random.seed(3537845168)


def load_config_procedures():
    """ Function to read the configurations from the specified config file """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='img_codec_config.json')
    config_file_str = parser.parse_args().config_file
    with open(str(config_file_str), 'r') as config:
        json_c = json.load(config)
    return json_c['run_spec'], json_c['autoencoder'], config_file_str


def main():
    if not platform.system().lower() == 'linux':
        raise RuntimeError('This code currently only works linux environments')
    else:
        os.environ['LD_LIBRARY_PATH'] = './jpeg2k_kakadu'
        os.environ['PATH'] = './jpeg2k_kakadu'

    run_spec, autoenc_spec, config_file_str = load_config_procedures()
    autoenc = AutoEnc(autoenc_spec, run_spec)

    # Copy config file to the execution folder
    if str(Path(config_file_str).parent) == '.':
        copyfile(config_file_str,
                 str(autoenc.out_name / config_file_str))

    if autoenc.run_cfg['generators']['train']['enabled']:
        autoenc.train_model()
    if autoenc.run_cfg['generators']['test']['enabled']:
        autoenc.test_model()


if __name__ == '__main__':
    main()
