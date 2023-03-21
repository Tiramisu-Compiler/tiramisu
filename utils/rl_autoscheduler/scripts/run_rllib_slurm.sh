#!/bin/bash
#SBATCH -p compute
#SBATCH --reservation c2
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=28
#SBATCH -t 7-0:00:00
#SBATCH -o outputs/job.train.out
#SBATCH -e outputs/job.train.err

. scripts/env.sh
. $CONDA_DIR/bin/activate
conda activate $CONDA_ENV

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
#nodes_array=dn[096,102-104]
nodes_array=($nodes)
head_node=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
ip_head=$ip_prefix:$PORT
echo "head node is at $ip_head"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip_prefix" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$ip_prefix"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  ip_prefix=${ADDR[1]}
else
  ip_prefix=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $ip_prefix"
fi


srun --nodes=1 --ntasks=1 -w $head_node ray start --num-cpus "${SLURM_CPUS_PER_TASK}" --head \
--node-ip-address="$ip_prefix" --port=$PORT --redis-password=$REDIS_PWD --block & 
sleep 10

echo "starting workers"
for ((  i=1; i<=$WORKER_NUM; i++ ))
do
    node2=${nodes_array[$i]}
    echo "i=${i}, w = ${w}, node2=$node2"
    srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus "${SLURM_CPUS_PER_TASK}" --address "$ip_head" --redis-password=$REDIS_PWD --block &
    sleep 5
done
python train_ppo.py --num-workers $(( $WORKER_PER_NODE * ( $WORKER_NUM + 1 )))
