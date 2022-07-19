#!/bin/bash
#SBATCH --job-name=embedNPMP
#SBATCH --mem=12000
#SBATCH -t 0-03:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -p olveczky,shared,cox,serial_requeue
#SBATCH --exclude=holy2c18111 #seasmicro25 was removed
#SBATCH --constraint="intel&avx2"
#SBATCH --output=/dev/null 
#SBATCH --error=/dev/null
source /n/home02/daldarondo/LabDir/Diego/bin/.customcommands.sh
setup_mujoco210_3.7
npmp_embed_single_batch "$@"
