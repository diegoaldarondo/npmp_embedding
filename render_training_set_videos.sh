#!/bin/bash
#SBATCH --job-name=renderVideo
#SBATCH --mem=6000
#SBATCH -t 0-00:10
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -p olveczky,shared,serial_requeue
#SBATCH --exclude=seasmicro25,holy2c18111
#SBATCH --constraint="intel&avx"
source /n/home02/daldarondo/LabDir/Diego/bin/.customcommands.sh
setup_mujoco200_3.7
render_training_set_single_batch "$@"