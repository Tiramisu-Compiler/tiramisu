#!/bin/sh
#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -t 7-0:00:00
#SBATCH -o outputs/job.%J.out
#SBATCH -e outputs/job.%J.err

source ~/.bashrc

conda activate tiramisu-build-envr

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
#nodes_array=dn[096,102-104]
nodes_array=($nodes)
head_node=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
port=6379
ip_head=$ip_prefix:$port
redis_password="ne2128"
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

export REDIS_PWD=$redis_password
export IP_HEAD=$ip_head
#tiramisu_path = '/scratch/nhh256/tiramisu/' # Put the path to your tiramisu installation here
export TIRAMISU_ROOT="/scratch/ne2128/tiramisu/"

srun --nodes=1 --ntasks=1 -w $head_node ray start --num-cpus "${SLURM_CPUS_PER_TASK}" --head \
--node-ip-address="$ip_prefix" --port=$port --redis-password=$redis_password --block & 
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))

echo "starting workers"
for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "i=${i}, node2=$node2"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus "${SLURM_CPUS_PER_TASK}" --address "$ip_head" --redis-password=$redis_password --block &
  sleep 5
done

python -u ppoCustomModelEnhancedHyper.py