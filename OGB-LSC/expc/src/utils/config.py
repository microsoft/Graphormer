# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
from easydict import EasyDict


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file')
    argparser.add_argument(
        '-i', '--id',
        metavar='I',
        default='',
        help='The commit id)')
    argparser.add_argument(
        '-t', '--ts',
        metavar='T',
        default='',
        help='The time stamp)')
    argparser.add_argument(
        '-d', '--dir',
        metavar='D',
        default='',
        help='The output directory)')
    args = argparser.parse_args()
    return args


def get_config_from_json(json_file):
    # parse the configurations from the configs json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(args):
    config = get_config_from_json(args.config)
    config.commit_id = args.id
    config.time_stamp = args.ts
    config.directory = args.dir
    return config

