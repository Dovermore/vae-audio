import pathlib
import os
import argparse
import json
import time
import numpy as np
from torchvision import transforms
import datasets as module_dataset
import transformers as module_transformer


def get_instance(module, name, config):
    """
    Get module indicated in config[name]['type'];
    If there are args to specify the module, specify in config[name]['args']
    """
    func_args = config[name]['args'] if 'args' in config[name] else None

    # if any argument specified in config[name]['args']
    if func_args:
        return getattr(module, config[name]['type'])(**func_args)
    # if not then just return the module
    return getattr(module, config[name]['type'])()


def save_json(x, fname, if_sort_key=False, n_indent=None):
    with open(fname, 'w') as outfile:
        json.dump(x, outfile, sort_keys=if_sort_key, indent=n_indent)


def main(config):
    """
    Audio procesing: the transformations and directories are specified by esc10_spectro1_config.json
    ---------------
    This parse the every entry of 'transform#' in esc10_spectro1_config.json,
    intialize the pytorch dataset object with the specified transforms,
    and save to the specified directory in esc10_spectro1_config.json.
    """
    # parse the transformers specified in esc10_spectro1_config.json
    list_transformers = [get_instance(module_transformer, i, config) for i in config if 'transform' in i]
    aggr_transform = transforms.Compose(list_transformers)
    config['dataset']['args']['transform'] = aggr_transform

    # get dataset and intialize with the parsed transformers
    dataset = get_instance(module_dataset, 'dataset', config)
    config['dataset']['args'].pop('transform', None)  # remove once dataset is intialized, in order to save json later

    # write config file to the specified directory
    processed_audio_savePath = os.path.join(os.path.expanduser(config['save_dir']), config['name'])
    if not os.path.exists(processed_audio_savePath):
        os.makedirs(processed_audio_savePath)
    print("Saving the processed audios in %s" % processed_audio_savePath)
    save_json(config, os.path.join(processed_audio_savePath, 'config.json'))

    # read, process (by transform functions in object dataset), and save
    start_time = time.time()
    for k in range(len(dataset)):
        audio_path = str(dataset.path_to_data[k])
        print("Transforming %d-th audio ... %s" % (k, audio_path))
        idx, y, x = dataset[k]

        if not os.path.exists(processed_audio_savePath):
            os.makedirs(processed_audio_savePath)
        np.save(os.path.join(processed_audio_savePath, pathlib.PurePath(dataset.path_to_data[k]).stem), x)

    print("Processing time: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Audio Transformation')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    config = json.load(open(args.parse_args().config))
    main(config)
