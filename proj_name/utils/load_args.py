import argparse
import configparser
import re


def load_args(config_path):
    """
    Parses command line arguments
    Returns a dictionary with command line and config file configs
    Gives priority to command line, can overwrite config file configs
    If the config file has numbers, they are returned as numbers and not as strings
    """
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)

    # Flags
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-no_wandb", action="store_true")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    with open(config_path) as f:
        config.read_file(f)
    configs = dict(config['configs'])

    def cast(val):
        '''
        Returns a float, int, bool, or string depending on what val is like
        '''
        float_rg = "(([+-]?([0-9]*)?[.][0-9]+)|([0-9]+[e][-][0-9]+))"
        if re.match(float_rg, val):
            return float(val)
        if val.isnumeric():
            return int(val)
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False
        return val

    configs = {k : cast(v) for k, v in configs.items()}

    for k, v in vars(args).items():
        if v is not None:
            configs[k] = v
    
    print("Configs used:", configs)

    return {**dict(configs), **vars(args)}