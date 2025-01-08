#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --mem-per-cpu=9000
#SBATCH --nodes=5
#SBATCH --time=24:00:00
#SBATCH -J fill_database_test
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

cd /home/s4307678/heston-moments-pdf
pip install -r requirements.txt

srun python simulation_database/fill_database.py
