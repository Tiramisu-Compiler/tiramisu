#!/bin/bash

for file in job_files/execute/execute_job*
do
  sbatch "$file"
done
