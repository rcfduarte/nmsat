#!/bin/bash
#$ -cwd
#$ -j y
#$ -q {{ queue }}.q
#$ -pe mpi {{ ppn }}
#$ -S /bin/bash

cd $SGE_O_WORKDIR
python {{ computation_script }}