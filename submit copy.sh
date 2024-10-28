pwd=$(pwd)
# cd ./load/nerf_synthetic/rotblender
# for case in $(find -mindepth 2 -maxdepth 2 -type d); do
#     echo "$case"
#     cd $pwd
#     jobname=$(basename $case)
#     sbatch --job-name=${jobname}_womask batch_train.sh configs/neus-rotblender-womask.yaml rotblender/$case neus_womask
#     sbatch --job-name=${jobname}_wmask batch_train.sh configs/neus-rotblender-wmask.yaml rotblender/$case neus_wmask
#     # sbatch batch_train.sh configs/rotneus-rotblender-wmask.yaml $case wmask_larger_region_more_sample
#     # sbatch batch_train.sh configs/rotneus-rotblender.yaml $case womask_larger_region_more_sample
# done

cd ./load/rot-zivid
for case in $(find -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do
    echo "$case"
    cd $pwd
    sbatch --job-name=$case batch_train.sh configs/neus-rotreal-womask.yaml $case neus_womask
    sbatch --job-name=${case}_wmask batch_train.sh configs/neus-rotreal-wmask.yaml $case neus_wmask
    # sbatch batch_train.sh configs/rotneus-rotblender-wmask.yaml $case wmask_larger_region_more_sample
    # sbatch batch_train.sh configs/rotneus-rotblender.yaml $case womask_larger_region_more_sample
done
