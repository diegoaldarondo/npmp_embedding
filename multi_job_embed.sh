#!/bin/bash
#SBATCH --job-name=embedNPMP
#SBATCH --mem=6000
#SBATCH -t 0-03:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -p olveczky,shared,serial_requeue
#SBATCH --exclude=holy2c18111 #seasmicro25 was removed
#SBATCH --constraint="intel&avx"
source /n/home02/daldarondo/LabDir/Diego/bin/.customcommands.sh
setup_mujoco200_3.7
npmp_embed_single_batch "$@"