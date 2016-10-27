#!/bin/bash
#SBATCH -N {{ nodes }}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ ppn }}
#SBATCH --time {{ walltime }}
#SBATCH --gres=mem256

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR
python {{ computation_script }}
