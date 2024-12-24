import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse


def setup():
    dist.init_process_group(
        backend='mpi'
    )

# --- 추론 함수 ---
def run_inference(world_size, model, dataloader):
    rank = 0
    """Distributed inference function"""
    # 모델 준비
    model = model.cuda()  # GPU에 모델 로드
    model = DDP(model, device_ids=[0])  # 분산 병렬화 설정

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
def main():
    """Main function for distributed inference"""
    # Get rank and world size from MPI
    print("Setup Start")
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = os.uname()[1]
    print(f"Running on host: {hostname} with rank {rank} out of {world_size}")



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
    results = run_inference(world_size, model, dataloader)

    # 결과 출력 (각 프로세스에서 실행된 데이터 수 확인)
    if rank == 0:  # 마스터 노드에서만 결과 확인
        print(f"Inference completed with results: {results}")
    cleanup()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes')
    args = parser.parse_args()
    world_size = args.world_size

    main()
