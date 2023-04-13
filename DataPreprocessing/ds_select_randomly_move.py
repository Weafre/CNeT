import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
import argparse
import shutil
from os import makedirs
from glob import glob
from tqdm import tqdm
import random as rn
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_select_randomly_move.py',
        description='split dataset into training and testing set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('n', help='Portion of random files to move.', type=float, default=1.0)

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    assert args.n > 0

    paths = glob(os.path.join(args.source, '**', f'*.ply'), recursive=True)
    paths = [x for x in paths if os.path.isfile(x)]
    files = [x[len(args.source) + 1:] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    assert len(files) > 0
    rn.shuffle(files)
    selection=rn.sample(range(0,files_len),int(files_len*args.n))
    files=[files[idx] for idx in selection]
    paths=[paths[idx] for idx in selection]
    files_with_paths = list(zip(files, paths))

    for file, path in tqdm(files_with_paths):
        target_path = os.path.join(args.dest, file)
        target_folder, _ = os.path.split(target_path)
        makedirs(target_folder, exist_ok=True)
        #shutil.copyfile(path, target_path)
        #os.symlink(path, target_path)
        shutil.move(path, target_path)

    no_file_in_origin = len(glob(os.path.join(args.source, '**', f'*.ply'), recursive=True))
    no_file_output = len(glob(os.path.join(target_folder, '**', f'*.ply'), recursive=True))
    logger.info(f'moved {no_file_output} models to {args.dest} leaving {no_file_in_origin} from {files_len}')