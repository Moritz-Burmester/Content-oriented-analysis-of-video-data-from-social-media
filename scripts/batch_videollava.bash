SBATCH --job-name=cideollava
SBATCH --ntasks-per-node=1
SBATCH --cpus-per-task=10
SBATCH --mem=30G
SBATCH --mail-type=BEGIN,END,FAIL
SBATCH --gres=gpu:1
SBATCH --partition=gpu-vram-12gb
SBATCH --time=12:00:00

conda activate videollava
../main.py