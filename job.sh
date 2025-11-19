#!/bin/bash
#SBATCH --job-name=model_compare       # Job name
#SBATCH --output=model_compare_%j.log   # Standard output and error log
#SBATCH --partition=gpu                  # Partition name (change if necessary)
#SBATCH --gres=gpu:1                     # Request one GPU
#SBATCH --mem=16G                        # Memory required (16GB as a safe margin)
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --nodes=1                        # Number of nodes

# Load necessary modules if required (example for Python)
module load python/3.11.7  # Adjust or remove based on your environment
module load conda/latest

# Activate virtual environment and run the job
cd /home/oschmerling_umass_edu/dysarthric-speech-recognition/
conda env create -f environment.yml -n cenv
python modelCompare.py --domain_a_dir "/home/oschmerling_umass_edu/UASpeech/Copy of UASpeech_original_C" --domain_b_dir "/home/oschmerling_umass_edu/UASpeech/Copy of UASpeech_original_FM"
