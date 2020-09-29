import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
import glob
from pandas import read_csv
from model.supervisor import EncoderDecoder


def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)


if __name__ == '__main__':
    # seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only',
                        default=False,
                        type=str,
                        help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file',
                        default="config/precip.yaml",
                        type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode',
                        default='ga_seq2seq',
                        type=str,
                        help='Run mode.')

    args = parser.parse_args()

    preprocessed_data_dir = './data/preprocessed_gauge_data/'
    start_index = len(preprocessed_data_dir)
    type_file = '.csv'
    end_index = -len(type_file)
    data_paths_preprocessed_data = glob.glob(preprocessed_data_dir + '*' +
                                             type_file)
    for index, file in enumerate(data_paths_preprocessed_data):
        file_name = file[start_index:end_index]
        file_name_list = file_name.split('_')
        gauge_name = file_name_list[0]
        dataset_npz = "data/npz/precip/{}.npz".format(gauge_name)
        dataset = read_csv(file, usecols=['precipitation'])
        np.savez(dataset_npz, precip_data=dataset)

        with open(args.config_file) as f:
            config = yaml.load(f)
        config['base_dir'] = "log/conv2d/{}/".format(gauge_name)
        config['data']['dataset'] = dataset_npz
        
        model = EncoderDecoder(is_training=True, **config)
        model.train()
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()

    if args.mode == 'train':
        model = EncoderDecoder(is_training=True, **config)
        model.train()
    elif args.mode == 'test':
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
