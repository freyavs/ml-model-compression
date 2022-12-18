#!/bin/bash -l
#PBS -N ml-compression
#PBS -l nodes=1:ppn=1
#PBS -l gpus=1
#PBS -l walltime=0:30:0

# module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
# module load pandas/1.1.2-foss-2020a-Python-3.8.2

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

pip install -r  requirements.txt
#
# copy input data from location where job was submitted from
cp $PBS_O_WORKDIR/* $TMPDIR
# cd $PBS_O_WORKDIR
# go to temporary working directory ( on local disk ) & run Python code

cd $TMPDIR

python experiments.py
# copy back output data , ensure unique filename using $PBS_JOBID
cp -r experiment_results $VSC_DATA/experiment_results
