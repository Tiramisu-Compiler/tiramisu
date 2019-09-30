#!/bin/bash

for file in execute/*
do
  sbatch "$file"
done
