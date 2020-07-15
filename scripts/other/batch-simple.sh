#!/bin/bash -l

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --partition=gpu    # Partition
#SBATCH --account=pa200702    # Replace with your system project
#SBATCH --gres=gpu:2		# For srun, allow access to 2 GPUs


which python
gcc --version
mpirun --version
nvcc --version
nvidia-smi
export PYTHONPATH="./:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1
# cd $HOME/kostis/GPU-BLonD


if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

source $HOME/.bashrc

$@


