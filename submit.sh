for case in $(ls ./load/nerf_synthetic); do
    sbatch batch_train.sh configs/neus-rotblender.yaml $case womask_larger_sample_region
    sbatch batch_train.sh configs/neus-rotblender-wmask.yaml $case wmask_larger_sample_region
    sbatch batch_train.sh configs/rotneus-rotblender-wmask.yaml $case wmask_larger_sample_region
    sbatch batch_train.sh configs/rotneus-rotblender.yaml $case womask_larger_sample_region
done
