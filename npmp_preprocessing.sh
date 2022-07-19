#!/bin/bash
#SBATCH --job-name=preprocessNPMP
# Job name
#SBATCH --mem=100000
# Job memory request
#SBATCH -t 0-10:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p olveczky,shared
#SBATCH --exclude=holy2c18111 #seasmicro25,
#SBATCH --constraint="intel&avx"
source ~/.bashrc
setup_mujoco210_3.7
stac_path=/n/home02/daldarondo/LabDir/Diego/tdata/dm/stac/results/long_clips/JDM31_day_8/total.p
save_path=/n/holylfs02/LABS/olveczky_lab/Diego/code/dm/npmp_embedding/JDM31_day_8.hdf5
npmp-preprocessing $stac_path $save_path
