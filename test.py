import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- 초기화 ---
def setup(rank, world_size, master_addr, master_port):
    """Initialize distributed process group"""
    dist.init_process_group(
        backend='gloo',  # NVIDIA GPU 간 최적화된 통신
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,       # 전체 rank
        world_size=world_size  # 전체 프로세스 수
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
        for data in dataloader:
            data = data.to(rank)  # 데이터도 GPU로 전송
            outputs = model(data)
            results.append(outputs)

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
    setup(rank, world_size, master_addr, master_port)

    # 간단한 모델 정의 (예: ResNet50)
    model = torch.hub.load('pytorch/vision:v1.11.0', 'resnet50', pretrained=False)

    # 가상의 데이터셋 (배치 크기: 32)
    dataloader = torch.utils.data.DataLoader(
        torch.randn(100, 3, 224, 224),  # 100개의 샘플, 이미지 크기 (3, 224, 224)
        batch_size=32
    )

    # 추론 실행
    results = run_inference(rank, world_size, model, dataloader)

    # 결과 출력 (각 프로세스에서 실행된 데이터 수 확인)
    if rank == 0:  # 마스터 노드에서만 결과 확인
        print(f"Inference completed with results: {results}")
    cleanup()

if __name__ == "__main__":
    import os
    import argparse
    from torch.multiprocessing import spawn

    # --- 설정 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--master_addr', type=str, default='192.168.100.131', help='Master node IP')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    args = parser.parse_args()

    # 전체 프로세스 수 계산
    world_size = args.nodes * args.gpus

    # 다중 프로세스 실행
    spawn(
        main,
        args=(world_size, args.master_addr, args.master_port),
        nprocs=args.gpus
    )
