#!/bin/bash

for file in wrappers/*
do
  sbatch "$file"
done
