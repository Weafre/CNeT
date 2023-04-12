import os
import argparse
import time
import gzip
import pickle
import torchac
import MinkowskiEngine as ME
import torch
import torch.nn as nn
import numpy as np

from utils.inout import  pmf_to_cdf, pc_transform_ycocg_n_partitioning
from Encoders.metadata_codec import save_compressed_file
from Training.cnet_attribute_training import ColorVoxelDNNTrainModule as YCoCgModel

print('Finished importing')

'''
CNeT attribute encoder
input: 
parser.add_argument("-level", '--octreedepth', type=int,
                    default=10,
                    help='depth of input octree to pass for encoder')
parser.add_argument("-depth", '--partitioningdepth', type=int,
                    default=3,
                    help='max depth to partition block')

parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
parser.add_argument("-output", '--outputpath', type=str, help='path to output files')

parser.add_argument("-color_voxeldnn_path", '--color_voxeldnn_path',action='append', type=str, help='path to input color model  checkpoint')
parser.add_argument("-signaling", '--signaling', type=str, help='special character for the output')
parser.add_argument("-usecuda", '--usecuda', type=bool,
                    default=True, action=argparse.BooleanOptionalAction,
                    help='using cuda or not')
Output: 
bitstream for the attribute and metadata are saved in Output/ folder
the bitrate (bpp) is caculated as (metadata file size + attribute file size)/number of input point

run ./command.sh to encode a list of input point clouds 
or run command below to encode a specific point cloud: 

python3 -m Encoders.cnet_attribute_encoder -level 10 -ply ply_path -output Output/   -color_voxeldnn_path checkpoint1 -color_voxeldnn_path checkpoint2 -color_voxeldnn_path checkpoint3  -signaling signal

with 
ply_path:  the file path of the point cloud
checkpoint1: checkpoint for predicting first channel 
checkpoint2: checkpoint of the model predicting 2nd channel from 1st channel
checkpoint3: checkpoint of the model predicting 3rd channel from 1st and 2nd channel
'''
def ColorVoxelDNN(args):
    global bbbits, ColorVoxelDNN, device
    bbbits=0
    pc_level, ply_path,output_path, color_voxeldnn_path ,signaling, usecuda= args
    block_bit_depth=8
    departition_level = pc_level - block_bit_depth
    sequence_name = os.path.split(ply_path)[1]
    sequence=os.path.splitext(sequence_name)[0]

    output_path = output_path+str(sequence)+'/'+signaling
    os.makedirs(output_path,exist_ok=True)
    occupancy_bin = output_path+'.occ.bin'
    color_bin = output_path+'.color.bin'
    metadata_file = output_path + '.metadata.bin'
    info = output_path +'.info.pkl'

    start = time.time()
    #getting encoding input data
    boxes, binstr, no_oc_voxels=pc_transform_ycocg_n_partitioning(ply_path,pc_level,departition_level)
    device = torch.device("cuda" if usecuda else "cpu")

    #restore ColorVoxelDNN
    ColorVoxelDNN=[]
    ymodel = YCoCgModel.load_from_checkpoint(color_voxeldnn_path[0], no_res=2, dim=0, lr=10, lrstep=1, gm=0.95)
    ymodel.eval()
    ymodel.freeze()
    ymodel.to(device)
    ColorVoxelDNN.append(ymodel)

    comodel = YCoCgModel.load_from_checkpoint(color_voxeldnn_path[1], no_res=2, dim=1, lr=10, lrstep=1, gm=0.95)
    comodel.eval()
    comodel.freeze()
    comodel.to(device)
    ColorVoxelDNN.append(comodel)

    cgmodel = YCoCgModel.load_from_checkpoint(color_voxeldnn_path[2], no_res=2, dim=2, lr=10, lrstep=1, gm=0.95)
    cgmodel.eval()
    cgmodel.freeze()
    cgmodel.to(device)
    ColorVoxelDNN.append(cgmodel)

    print('Checkpoints loaded....')

    loadtime=time.time()
    print("Loading time: ", loadtime-start)

    #encoding function
    flags=[]
    print("Encoding: ",len(boxes), ' blocks')
    with  open(color_bin , 'wb') as colorbit:
        encoding_executer(boxes, colorbit,flags)

    with gzip.open(metadata_file, "wb") as f:
        ret = save_compressed_file(binstr, pc_level, departition_level)
        f.write(ret)

    print('Encoded file: ', ply_path)
    end = time.time()
    print('Encoding time: ', end - start)

    color_size= int(os.stat(color_bin).st_size) * 8
    metadata_size = int(os.stat(metadata_file).st_size) * 8 + len(flags)*2+len(boxes)*36
    avg_bpov = (color_size + metadata_size) / no_oc_voxels

    print('Checkpoints: ',color_voxeldnn_path)
    print('Occupied Voxels: %04d' % no_oc_voxels)
    print('Color bitstream: ', color_bin)
    print('Metadata bitstream', metadata_file )
    print('Encoding information: ', info)
    print('Metadata and file size(in bits): ', metadata_size, color_size)
    print('Average bits per occupied voxels: %.04f' % avg_bpov)


def encoding_executer(boxes, color_bits, flags):
    no_bits=0
    for dimension in range(3):
        no_bits = 0
        for i in range(len(boxes)):
            bits= encode_color_block(boxes[i], color_bits, dimension)

def encode_color_block(block, bitstream=None, dimension=0):

    global bbbits, ColorVoxelDNN, device
    coord=block[:,:3]
    feats=block[:,3:].astype(np.float32)

    curr_feat=torch.tensor(feats[:, :(dimension + 1)]).to(device)
    coords, feats = ME.utils.sparse_collate([coord, ], [feats,])
    sparse_inputs = ME.SparseTensor(curr_feat, coords, device=device)
    if(dimension==0):
        predicts_features=ColorVoxelDNN[dimension](sparse_inputs, None)
        true_features=torch.round((feats[:, int(dimension)] * 255.5 + 255.5 - 127.5))

    elif (dimension == 1 or dimension==2):
        r_feats = feats[:, :dimension]
        sparse_rg_inputs = ME.SparseTensor(r_feats, coords, device=device)
        predicts_features = ColorVoxelDNN[dimension](sparse_inputs, sparse_rg_inputs)
        true_features = torch.round((feats[:, int(dimension)] * 255.5 + 255.5))

    probs = torch.softmax(predicts_features, dim=1).detach().clone().cpu()  # nx256
    cdf_probs = pmf_to_cdf(probs)


    channel = true_features.detach().clone().cpu()
    channel = channel.type(torch.int16)
    byte_stream=torchac.encode_float_cdf(cdf_probs, channel, check_input_bounds=True)
    total_bits = len(byte_stream) * 8
    if (bitstream != None):
        bitstream.write(byte_stream)
    return total_bits

# Main launcher
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')
    parser.add_argument("-depth", '--partitioningdepth', type=int,
                        default=3,
                        help='max depth to partition block')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-output", '--outputpath', type=str, help='path to output files')

    parser.add_argument("-color_voxeldnn_path", '--color_voxeldnn_path',action='append', type=str, help='path to input color model  checkpoint')
    parser.add_argument("-signaling", '--signaling', type=str, help='special character for the output')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True, action=argparse.BooleanOptionalAction,
                        help='using cuda or not')
    args = parser.parse_args()
    ColorVoxelDNN([args.octreedepth, args.plypath,args.outputpath, args.color_voxeldnn_path,args.signaling, args.usecuda])
    # python3 -m Encoders.cnet_attribute_encoder -level 10 -ply "../Datasets/TestPCs/10bits/phil_0010.ply" -output Output/   -color_voxeldnn_path "../ColorVoxelDNN/Model/ColorVoxelDNN/jn-0505-ycocg-y/best_val_checkpoint_model__lr_15_b_16_da_8_nores_2_nofil_256_nomix_10_schedule_20.95_dim_0-epoch=247-val_loss=1.87.ckpt" -color_voxeldnn_path "../ColorVoxelDNN/Model/ColorVoxelDNN/jn-0505-ycocg-co/best_val_checkpoint_model__lr_15_b_4_da_8_nores_2_nofil_256_nomix_10_schedule_20.95_dim_1-epoch=163-val_loss=1.33.ckpt" -color_voxeldnn_path "../ColorVoxelDNN/Model/ColorVoxelDNN/jn-0505-ycocg-cg/best_val_checkpoint_model__lr_15_b_4_da_8_nores_2_nofil_256_nomix_10_schedule_20.95_dim_2-epoch=116-val_loss=0.81.ckpt"  -signaling signal