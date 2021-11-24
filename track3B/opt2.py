import argparse
import os
import yaml
from pprint import pprint
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Novelty')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config/coco_full.yaml')

    parser.add_argument('--output',  '-o',
                        dest="output",
                        help =  'path to outputs',
                        default='')

    parser.add_argument('--model', help='model name',default='gcn4layer_fc')
    parser.add_argument('--modelpath',type=str,help='checkpoint of the model')
    parser.add_argument('--testrun', action='store_true', help='test run with few dataset')


    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc, flush=True)

    #other command line (CLI) args .. seperate from config file
    extra = {}
    extra['output'] = args.output #keep other args in there
    extra['model'] = args.model
    extra['modelpath'] = args.modelpath
    extra['testrun'] = args.testrun
    config['CLI'] = extra
    return config

if __name__ == '__main__':
    # parse command line arguments for the config file
    config = parse_args()
    pprint('parsed input parameters:')
    pprint(config)

    # # Make result folders if they do not exist
    results_dir = os.path.join(config['CLI']['output'], 'results')
    os.makedirs(results_dir,exist_ok=True)

    print(config)
    timestr = time.strftime("%Y-%m-%d.%H:%M:%S")
    name_timestamp = "_exp_" + timestr
    # Save the config file to results folder:
    with open(os.path.join(results_dir, f'config{name_timestamp}.yaml'), 'w') as f:
        f.write(yaml.dump(config))



