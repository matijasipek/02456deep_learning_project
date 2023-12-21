#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT
# # SBATCH --job-name=per_sample
# #SBATCH --output=cifar_per_sample-%J.out
# #SBATCH --cpus-per-task=2
# #SBATCH --time=12:00:00
# #SBATCH --mem=42gb
# #SBATCH --nodes=1
# #SBATCH --ntasks=10
# #SBATCH --gres=gpu:3
# #SBATCH --mail-user=blia@dtu.dk
# #SBATCH --mail-type=END,FAIL
# #SBATCH --export=ALL
# FD
## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"


lr_group="0.1"
n_clients=10 
split=non_iid 
local_epoch=1
method=check_zeta
non_iid_alpha=0.1 
dataset=wind 
model_type=m_cnn 
version=2
num_rounds=50
sigma=0
start_round=0
start_client=0
end_client=9
partition_type="non_iid"
num2=3
num3=6

# echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}

for s_lr in $lr_group # 0.1 0.01 0.001 -> learning rates, although we only use 0.1
do
    for round in $(seq "$start_round" 1 "$num_rounds") # 0 1 2... -> training rounds
    do
        for i in $(seq "$start_client" 1 "$end_client") # 0 1 2 3 4 5 6 7 8 9 -> client index
        do
            python3 train_wind_models.py --n_clients "$n_clients" --split "$split" --sigma "$sigma" --num_local_epochs "$local_epoch" \
                --method "$method" --version "$version" --lr "$s_lr" \
                --num_rounds "$num_rounds" --use_local_id "$i" --dataset "$dataset" --opt client \
                --model_type "$model_type" --non_iid_alpha "$non_iid_alpha" --partition_type "$partition_type" --start_round "$start_round" --round "$round" & ### & -> for parallel processing of clients
        done
        wait # -> wait for all clients to finish training withing a training round
    done 

done 

