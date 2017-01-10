#!/bin/bash
#SBATCH -N {{ nodes }}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ ppn }}
#SBATCH --time {{ walltime }}
#SBATCH --mem={{ mem }}

module load pystuff
module load mpi/openmpi/1.10.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR
python {{ computation_script }}
