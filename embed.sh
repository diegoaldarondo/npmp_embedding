#!/bin/bash
#SBATCH --job-name=handler
#SBATCH --mem=30000
#SBATCH -t 0-05:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky,shared
#SBATCH --exclude=holy2c18111 #seasmicro25 was removed
#SBATCH --constraint="intel&avx"
set -e
source /n/home02/daldarondo/LabDir/Diego/bin/.customcommands.sh
setup_mujoco200_3.7
bash -c "$1"
bash -c "$2"
