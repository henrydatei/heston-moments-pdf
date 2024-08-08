#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=9000
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -J gcd_solve_for_s
#SBATCH --error="/home/s4307678/.out/myjob-%J.out"
#SBATCH --output="/home/s4307678/.out/myjob-%J.out"

# Directory path
DIR="/home/s4307678/.out/"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
  # If the directory does not exist, create it
  mkdir -p "$DIR"
  echo "Directory $DIR created."
else
  echo "Directory $DIR already exists."
fi

ml purge 

ml release/24.04 Mathematica/13.0.1

srun math -run /home/s4307678/gcd_solve_for_s.wls
