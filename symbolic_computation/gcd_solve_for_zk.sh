#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=9000
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -J gcd_solve_for_zk
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

ml release/24.04 GCCcore/13.2.0 Python/3.11.5

pip install sympy

srun python /home/s4307678/gcd_solve_for_zk.py
