#!/bin/sh

#SBATCH --mem-per-cpu=4000
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -J fill_database_with_distances
#SBATCH --array=0-99
#SBATCH --error="/home/s4307678/.out/myjob-%A_%a.out"
#SBATCH --output="/home/s4307678/.out/myjob-%A_%a.out"

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

ml --force purge 

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

srun python compare_distributions/fill_database_with_distances.py --i $SLURM_ARRAY_TASK_ID --chunks 100

deactivate