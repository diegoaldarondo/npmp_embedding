#!/bin/bash
#SBATCH --job-name=submit_noise_analysis
# Job name
#SBATCH --mem=2000
# Job memory request
#SBATCH -t 1-00:00
# Time limit hrs:min:sec
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky,shared
source ~/.bashrc
setup_mujoco210_3.7
python -c "import submit_noise_analysis; submit_noise_analysis.submit(\"$1\")"
wait
