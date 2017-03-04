#!/bin/bash
#$ -cwd
#$ -j y
#$ -q {{ queue }}.q
#$ -l slots={{ ppn }}
#$ -S /bin/bash

cd $SGE_O_WORKDIR
python {{ computation_script }}