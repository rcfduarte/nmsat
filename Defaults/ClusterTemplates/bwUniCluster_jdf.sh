#!/bin/sh
#MSUB -d {{ script_folder }}
#MSUB -l walltime={{ walltime }}
#MSUB -l nodes={{ nodes }}:ppn={{ ppn }}
#MSUB -l mem={{ mem }}GB
#MSUB -q {{ queue }}

export OMP_NUM_THREADS=${MOAB_PROCCOUNT}

export PATH=/home/fr/fr_fr/fr_rd1000/NEST-root/NEST_2.6.0/nest_install/bin/:$PATH
export PYTHONPATH=/home/fr/fr_fr/fr_rd1000/NEST-root/NEST_2.6.0/nest_install/lib/python2.7/site-packages/:$PYTHONPATH
export PYTHONPATH=/home/fr/fr_fr/fr_rd1000/NEST-root/NEST_2.6.0/nest_install/lib/python2.7/dist-packages/:$PYTHONPATH
export PYTHONPATH=/home/fr/fr_fr/fr_rd1000/sklearn/scikit-learn-0.16.1_install/lib/python2.7/site-packages/:$PYTHONPATH
export PYTHONPATH=/home/fr/fr_fr/fr_rd1000/NetworkSimulationTestbed/:$PYTHONPATH

time \
    python \
        {{ computation_script }}