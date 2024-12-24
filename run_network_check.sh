export MASTER_ADDR="192.168.100.131"
export MASTER_PORT=29500
export WORLD_SIZE=2
export LOCAL_RANK=0
if [ "$(hostname)" = "master" ]; then
    export WORLD_RANK=0
elif [ "$(hostname)" = "soda1" ]; then
    export WORLD_RANK=1
elif [ "$(hostname)" = "soda2" ]; then
    export WORLD_RANK=2
elif [ "$(hostname)" = "soda3" ]; then
    export WORLD_RANK=3
fi
python3 network_check.py --local_rank=$LOCAL_RANK --backend=gloo