import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size, master_addr, master_port):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

# --- 추론 함수 ---
def run_inference(rank, world_size, model, dataloader):
    """Distributed inference function"""
    # 모델 준비
    model = model.to(rank)  # GPU에 모델 로드
    model = DDP(model, device_ids=[rank])  # 분산 병렬화 설정

    # 추론 수행
    model.eval()
    with torch.no_grad():
        results = []
        total_iters = len(dataloader)
        for i, data in enumerate(dataloader):
            data = data.to(rank)  # 데이터도 GPU로 전송
            outputs = model(data)
            results.append(outputs)
            if rank == 0:  # 마스터 노드에서만 출력
                print(f"Remaining iterations: {total_iters - (i+1)}/{total_iters}")

        # 결과를 모든 프로세스로부터 수집
        gathered_results = [torch.zeros_like(results[0]) for _ in range(world_size)]
        dist.all_gather(gathered_results, results[0])  # 모든 프로세스에서 데이터 수집
        return gathered_results

# --- 종료 ---
def cleanup():
    """Clean up distributed environment"""
    dist.destroy_process_group()

# --- 메인 ---
def main(rank, world_size, master_addr, master_port):
    """Main function for distributed inference"""
    print("Setup Start")
    setup(rank, world_size, master_addr, master_port)

    # 간단한 모델 정의 (예: ResNet50)
    print("Model Load Start")
    model = torch.hub.load('pytorch/vision:v1.11.0', 'resnet18', pretrained=False)

    # 가상의 데이터셋 (배치 크기: 32)
    print("Data Load Start")
    dataloader = torch.utils.data.DataLoader(
        torch.randn(40, 3, 224, 224),  # 100개의 샘플, 이미지 크기 (3, 224, 224)
        batch_size=4
    )

    # 추론 실행
    print("Inference Start")
    results = run_inference(rank, world_size, model, dataloader)

    # 결과 출력 (각 프로세스에서 실행된 데이터 수 확인)
    if rank == 0:  # 마스터 노드에서만 결과 확인
        print(f"Inference completed with results: {results}")
    cleanup()

if __name__ == "__main__":
    import os
    import argparse
    from torch.multiprocessing import spawn

    # Get computer's hostname
    hostname = os.uname().nodename
    print(f"Running on host: {hostname}")
    if(hostname == 'master'):
        rank = 0
    elif(hostname == 'soda1'):
        rank = 1
    elif(hostname == 'soda2'):
        rank = 2
    elif(hostname == 'soda3'):
        rank = 3

    # --- 설정 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--master_addr', type=str, default='192.168.100.131', help='Master node IP')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    args = parser.parse_args()

    # 전체 프로세스 수 계산
    world_size = args.nodes * args.gpus

    print(f"Rank: {rank}, World Size: {world_size}, Master Address: {args.master_addr}, Master Port: {args.master_port}")
    main(rank, world_size, args.master_addr, args.master_port)
    # 다중 프로세스 실행
    # spawn(
    #     main,
    #     args=(rank, world_size, args.master_addr, args.master_port),
    #     nprocs=args.gpus
    # )
