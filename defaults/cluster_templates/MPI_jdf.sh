#!/bin/bash
#$ -cwd
#$ -o /data/corpora/sge2/nbl/rendua/
#$ -e /data/corpora/sge2/nbl/rendua/
#$ -q {{ queue }}.q
#$ -S /bin/bash

cd $SGE_O_WORKDIR
python {{ computation_script }}