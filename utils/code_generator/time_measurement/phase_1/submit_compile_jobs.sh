#!/bin/bash

for file in job_files/compile_job*
do
  sbatch "$file"
done
