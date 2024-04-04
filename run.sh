#!/bin/bash
for case in `ls ./load/nerf_synthetic/`; do
    echo $case
    python launch.py --config configs/neus-rotblender-wmask.yaml --gpu 0 --train dataset.scene="$case" tag=neus-wmask &&
    python launch.py --config configs/neus-rotblender.yaml --gpu 0 --train dataset.scene="$case" tag=neus-womask &&
    python launch.py --config configs/rotneus-blender-wmask.yaml --gpu 0 --train dataset.scene="$case" tag=rotneus-wmask &&
    python launch.py --config configs/rotneus-blender.yaml --gpu 0 --train dataset.scene="$case" tag=rotneus-womask

    # python launch.py --config configs/rot-neus-blender.yaml --gpu 0 --train dataset.scene="$case" dataset.apply_mask=true tag=wmask-all
done
