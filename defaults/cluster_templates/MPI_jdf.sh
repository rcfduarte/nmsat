#!/bin/bash
#$ -cwd
#$ -j y
#$ -q {{ queue }}.q
#$ -pe mpich {{ ppn }}
#$ -l h_rt={{ walltime }}
#$ -l mem_free={{ mem }}
#$ -S /bin/bash

binary=/usr/local/bin/Rscript
cd $SGE_O_WORKDIR
#$binary grid-demo.r
python {{ computation_script }}