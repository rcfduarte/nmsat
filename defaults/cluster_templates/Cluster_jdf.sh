#!/bin/bash
# Example script to generate scripts to be run on a cluster. This script should not called directly as it only serves
# as a template; some values will be replaced by NMT during export. For more details see the NMT User Manual.

# NOTE: this example is specific for a SLURM queueing system.

#SBATCH -N {{ nodes }}
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node={{ ppn }}
# #SBATCH --cpus-per-task={{ ppn }}
#SBATCH --time {{ walltime }}
#SBATCH --mem={{ mem }}

# load all required modules here
module load pystuff
module load mpi/openmpi/1.10.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


cd $SLURM_SUBMIT_DIR
# do not modify the following line!
python {{ computation_script }}
