#!/bin/bash
#SBATCH --output=/scratch/niemine8/vgs/data-annotation/log/yc2_threshold.out
#SBATCH --error=/scratch/niemine8/vgs/data-annotation/log/yc2_threshold.err
#SBATCH -J yc2thr
#SBATCH --mail-user=elias.nieminen@tuni.fi
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu --gres=gpu:1
#SBATCH --time=38:00:00
#SBATCH --mem=16000
echo "Starting job"
source activate tf250  # Replace dl-test with your environment name
echo "Loaded"
export HDF5_USE_FILE_LOCKING='FALSE'
python /scratch/niemine8/vgs/data-annotation/sn_threshold.py  # Replace the path to the script with your own.