#!/bin/bash
###

#SBATCH --job-name=accelerate-example   # Job name
#SBATCH --nodes=2                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --gres=gpu:8                    # Number of GPUs per node
#SBATCH --cpus-per-task=128             # Number of CPUs per task
#SBATCH --mem=0                         # Memory per node (0 for all)
#SBATCH --time=00:30:00                 # Time limit
#SBATCH --output="%x-%j.out"            # Output log file
#SBATCH --exclusive                     # Exclusive node access
#SBATCH --export=ALL                    # Export all environment variables

# Activate the virtualenv with installed `accelerate` and other dependencies
source .env/bin/activate

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
# export NCCL_DEBUG=INFO

# use cached dataset and model
export HF_HUB_OFFLINE=1

# Setup `accelerate` launch args
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
GPUS_PER_NODE=8
NNODES=2
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Run the multinode training
echo "START TIME: $(date)"
echo "MASTER_ADDR ${head_node_ip}"

srun \
    accelerate launch \
    --config_file ./fsdp_config.yaml \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    complete_nlp_example.py \
    --mixed_precision fp16 \
    --output_dir ./output

echo "END TIME: $(date)"
