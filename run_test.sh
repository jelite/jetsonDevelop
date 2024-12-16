export MASTER_ADDR="master_node_ip"
export MASTER_PORT="29500"
export RANK=0
export WORLD_SIZE=4
export LOCAL_RANK=0
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT inference.py
