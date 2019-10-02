#!/bin/bash

for file in job_files/wrappers/wrappers_job*
do
  sbatch "$file"
done
