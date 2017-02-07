#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time 01-00:00:00
#SBATCH --mem=48000

module load pystuff
module load mpi/openmpi/1.10.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR
python /home/neuro/Desktop/CODE/network_simulation_testbed/export/ED_spikepatterninput_training_parameters/ED_spikepatterninput_training_parameters_lexicon_size=110_T=100.py
