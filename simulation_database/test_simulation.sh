#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --mem-per-cpu=9000
#SBATCH --nodes=70
#SBATCH --time=36:00:00
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

python -m venv venv
source venv/bin/activate

pip install matplotlib
pip install numpy
pip install pandas
pip install scipy
# pip install seaborn
# pip install sympy
pip install tqdm
# pip install yfinance
# pip install pytorch-lightning

srun python simulation_database/fill_database.py

deactivate