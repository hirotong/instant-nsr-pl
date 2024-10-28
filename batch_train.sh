#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20gb
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1    # Number of GPUs (per node)
#SBATCH --account=R-20767-01
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

##SBATCH --job-name=neus
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jinguang.tong@csiro.au

#This is a comment
#Anything above this line starting with #SBATCH is a set of instructions for the batch system e.g. how much memory you want.
#Do NOT modify "nodes=1" unless you know what you're doing - most "desktop" programs can't use more than one node

#The below is what you want to actually run/do

echo "number of cores is $SLURM_NTASKS"
echo "job name is $SLURM_JOB_NAME"
module load miniconda3 cuda
conda activate neus
conf=$1
case=$2
tag=$3
echo "case name $case, using configuration $conf"
# python train.py --case $case --conf $conf &&
python launch.py --config "$conf" --gpu 0 --train dataset.scene="$case" tag=$tag
# python train.py --mode valid_mesh --case $case --conf $conf --resolution 512
sleep 120
