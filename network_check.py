import torch.distributed as dist
import os

def setup(rank, world_size):
    dist.init_process_group(
        backend="gloo",  # Use "nccl" if using GPUs
        init_method="tcp://192.168.100.131:29500",
        rank=rank,
        world_size=world_size,
    )
    print("Master node initialized at tcp://192.168.100.131:29500")
    
def main():
    setup(rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))  # Master node is rank 0

if __name__ == "__main__":
    main()

# nc -zv 192.168.100.131 29500