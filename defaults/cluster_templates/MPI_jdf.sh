#!/bin/bash
#$ -cwd
#$ -j y
#$ -q {{ queue }}.q
#$ -pe mpich {{ ppn }}
#$ -S /bin/bash

cd $SGE_O_WORKDIR
python {{ computation_script }}