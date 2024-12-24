import torch.distributed as dist
import torch
import os


def main():
    # Initialize the distributed process group
    dist.init_process_group(
        backend="mpi",  # Use the MPI backend
    )
    
    # Get the rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Get hostname
    hostname = os.uname()[1]
    print(f"Running on host: {hostname}")
    print(f"Hello from rank {rank} out of {world_size} processes")

    # Example: All-reduce operation
    tensor = torch.tensor([rank], dtype=torch.float32)
    print(f"Before all_reduce, rank {rank}: {tensor.item()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce, rank {rank}: {tensor.item()}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
