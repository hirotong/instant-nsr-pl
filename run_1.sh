#!/bin/bash
pwd=$(pwd) 
cd ./load/rot-blender-tat
masks=(0)
cases=$(find -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
# cases=("dmask" "dragon")
for case in ${cases[@]}; do
    echo $case
    for mask in ${masks[@]}; do
        cd $pwd
        python launch.py --config configs/neus-rotblender-tat-wmask.yaml --gpu 1 --train dataset.scene="$case" tag=neus-wmask_${mask} dataset.mask_level=${mask} 
        # python launch.py --config configs/neus-rotblender-tat-womask.yaml --gpu 2 --train dataset.scene="$case" tag=neus-womask 
    done
    # python launch.py --config configs/neus-rotblender-wmask.yaml --gpu 0 --train dataset.scene="$case" tag=neus-wmask &&
    # python launch.py --config configs/neus-rotblender.yaml --gpu 0 --train dataset.scene="$case" tag=neus-womask &&
    # python launch.py --config configs/rotneus-blender-wmask.yaml --gpu 0 --train dataset.scene="$case" tag=rotneus-wmask &&
    # python launch.py --config configs/rotneus-blender.yaml --gpu 0 --train dataset.scene="$case" tag=rotneus-womask

    # python launch.py --config configs/rot-neus-blender.yaml --gpu 0 --train dataset.scene="$case" dataset.apply_mask=true tag=wmask-all
done

