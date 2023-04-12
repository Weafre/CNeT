#!/bin/bash

pc_level=10
echo "What is signaling text?"
read signaling

echo "What is the command type? encoding color attribute: 1 "
read command_to_run

pcs=(   "../Datasets/TestPCs/10bits/phil_0010.ply"  "../Datasets/TestPCs/10bits/ricardo_0010.ply"  "../Datasets/TestPCs/TCSVT_CNeT/10bits/redandblack_vox10_1550.ply"  "../Datasets/TestPCs/TCSVT_CNeT/10bits/loot_vox10_1200.ply"  "../Datasets/TestPCs/10bits/MPEG_Thaidancer_vox10.ply"   "../Datasets/TestPCs/10bits/MPEG_boxer_vox10.ply"  "../Datasets/TestPCs/10bits/MPEG_CAT1_Frog_00067_vox10.ply"  "../Datasets/TestPCs/10bits/MPEG_CAT1_Arco_Valentino_Dense_vox10.ply"  "../Datasets/TestPCs/10bits/MPEG_CAT1_Shiva_00035_vox10.ply"  "../Datasets/TestPCs/10bits/PAULO_bumbameuboi.ply"  "../Datasets/TestPCs/10bits/PAULO_romanoillamp.ply" )

models=("Model/jn-0505-ycocg-y/best_checkpoint.ckpt" "Model/jn-0505-ycocg-co/best_checkpoint.ckpt" "Model/jn-0505-ycocg-cg/best_checkpoint.ckpt")


case "$command_to_run" in


    "1")
      for pc in "${pcs[@]}";
      do
      python3 -m Encoders.cnet_attribute_encoder -level 10 -ply $pc -output Output/   -color_voxeldnn_path ${models[0]} -color_voxeldnn_path ${models[1]} -color_voxeldnn_path ${models[2]}  -signaling "${signaling}"
      done
    ;;
esac
