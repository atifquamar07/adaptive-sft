#!/bin/bash
#SBATCH --job-name=ministral
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err


mkdir -p logs

echo "=== SLURM placement ==="
echo "JobID=$SLURM_JOB_ID"
echo "NodeList=$SLURM_JOB_NODELIST"
echo "Host=$(hostname -s)"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "======================="

nvidia-smi -L
nvidia-smi

conda activate adaptive-sft
bash scripts/run_all_multi_gpu.sh smoke
