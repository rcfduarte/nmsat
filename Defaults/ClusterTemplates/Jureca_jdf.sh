#!/bin/bash
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks-per-node={{ ppn }}
#SBATCH --time={{ walltime }}
#SBATCH --partition={{ queue }}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR
python {{ computation_script }}
