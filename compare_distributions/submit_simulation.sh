#!/bin/sh

#SBATCH --mem-per-cpu=8000
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -J fill_database_with_distances
#SBATCH --array=0-29
#SBATCH --error="/home/s4307678/.out/myjob-%A_%a.out"
#SBATCH --output="/home/s4307678/.out/myjob-%A_%a.out"

# Wait for a few seconds before starting the job
sleep $((SLURM_ARRAY_TASK_ID*30))

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

ml release/24.04 slurm GCCcore/13.2.0 Python/3.11.5

cd /home/s4307678/heston-moments-pdf

python -m venv venv
source venv/bin/activate

pip install --no-cache-dir matplotlib
pip install --no-cache-dir numpy
pip install --no-cache-dir pandas
pip install --no-cache-dir scipy
# pip install seaborn
# pip install sympy
pip install --no-cache-dir tqdm
# pip install yfinance
# pip install pytorch-lightning

srun python compare_distributions/fill_database_with_distances.py --i $SLURM_ARRAY_TASK_ID --chunks 30

deactivate