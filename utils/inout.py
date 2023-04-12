import numpy as np
import time
from glob import glob
import multiprocessing
from tqdm import tqdm
from pyntcloud import PyntCloud
import pandas as pd
import torch
import os

from DataPreprocessing.training_data_pipeline import transformRGBToYCgCoR
from utils.octree_partition import partition_octree

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

def get_bin_stream_blocks(path_to_ply, pc_level, departition_level):
    # co 10 level --> binstr of 10 level, blocks size =1
    level = int(departition_level)
    pc = PyntCloud.from_file(path_to_ply)
    points = pc.points.values
    no_oc_voxels = len(points)
    box = int(2 ** pc_level)
    blocks2, binstr2 = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    return no_oc_voxels, blocks2, binstr2

def occupancy_map_explore(ply_path, pc_level, departition_level):
    no_oc_voxels, blocks, binstr = get_bin_stream_blocks(ply_path, pc_level, departition_level)
    return blocks, binstr

def pc_transform_ycocg_n_partitioning(ply_path, pc_level, departition_level):
    level = int(departition_level)
    pc = PyntCloud.from_file(ply_path)

    try:
        cols=['x', 'y', 'z','red', 'green', 'blue']
        points=pc.points[cols].values

    except:
        cols = ['x', 'y', 'z', 'r', 'g', 'b']
        points = pc.points[cols].values
    nopoints=points.shape[0]
    points = np.round(points)
    color = points[:, 3:].astype(np.int16)
    color = transformRGBToYCgCoR(8, color)
    color = color.astype(np.float32)
    color[:, 0] += 127.5

    step = (pow(2,9) - 1.) / 2.
    color = (color - step) / step
    points[:, 3:] = color
    box = int(2 ** pc_level)
    blocks, binstr = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    return blocks, binstr, nopoints

def pmf_to_cdf(pmf):
  cdf = pmf.cumsum(dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  cdf_with_0 = cdf_with_0.clamp(max=1.)
  return cdf_with_0