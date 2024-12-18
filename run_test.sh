export MASTER_ADDR="192.168.100.131"
export MASTER_PORT=29500
if [ "$(hostname)" = "master" ]; then
    export RANK=0
elif [ "$(hostname)" = "soda1" ]; then
    export RANK=1
elif [ "$(hostname)" = "soda2" ]; then
    export RANK=2
elif [ "$(hostname)" = "soda3" ]; then
    export RANK=3
fi
echo $RANK
export WORLD_SIZE=4
python3 test.py
