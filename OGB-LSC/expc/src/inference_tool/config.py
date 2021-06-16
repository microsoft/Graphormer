# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
from easydict import EasyDict


def get_config_from_json(json_file):
    # parse the configurations from the configs json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(args):
    config = get_config_from_json(args[1])
    config.dataset = args[2]
    config.checkpoint = args[3]
    config.output = args[4]
    return config

