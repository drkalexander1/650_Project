#!/bin/bash

#"#SBATCH" directives that convey submission options:
#SBATCH --job-name=Project_Midpoint
#SBATCH --account=si650f25s101_class
#SBATCH --partition=standard
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=output_corpus.txt
#SBATCH --error=error_corpus.txt

# Change to the directory where the script is located
cd $SLURM_SUBMIT_DIR || cd /home/drkalex/650-project/650_Project

# Print current directory for debugging
echo "Working directory: $(pwd)"
echo "Contents of corpus directory:"
ls -la corpus/ 2>&1 || echo "corpus directory not found"

# The application(s) to execute along with its input arguments and options:
/bin/hostname
# module load python3.11-anaconda/2024.02
source venv/bin/activate
# Use explicit path to venv python to ensure correct environment
./venv/bin/python compute_pairwise_similarities.py