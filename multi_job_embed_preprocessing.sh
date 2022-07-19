#!/bin/bash
#SBATCH --job-name=preprocessNPMP
# Job name
#SBATCH --mem=5000
# Job memory request
#SBATCH -t 0-00:10
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky,shared,serial_requeue
#SBATCH --exclude=holy2c18111 #seasmicro25 was removed
#SBATCH --constraint="intel&avx2"
source ~/.bashrc
setup_mujoco210_3.7
npmp-preprocessing-single-batch