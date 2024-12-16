import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 환경변수로부터 분산 설정 읽기
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    # 분산 환경 초기화
    dist.init_process_group(
        backend='nccl', 
        init_method=f'env://',  # 기본적으로 env://면 MASTER_ADDR, MASTER_PORT 사용
        world_size=world_size, 
        rank=rank
    )
    
    # 현재 노드(컴퓨터)에서 사용 가능한 디바이스 설정
    # 일반적으로 한 프로세스당 하나의 GPU를 할당하는 식으로 구현하나,
    # 상황에 따라 조정 가능
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 모델 준비
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # DDP 래핑
    # DDP는 일반적으로 single-process single-GPU 실행을 가정
    # 즉, torchrun 등으로 GPU당 하나의 프로세스 실행 시 각 프로세스 하나의 GPU 할당
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 더미 입력 (각 노드에서 batch의 일부를 처리)
    # 예: 총 batch_size=128, world_size=4라면 각 노드당 32개 처리
    batch_size_per_node = 32
    input_data = torch.randn(batch_size_per_node, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(input_data)
    print(f"Rank {rank} output shape: {outputs.shape}")
    
    # 모든 노드에서 inference 결과를 gather하거나 별도 처리 가능
    # 분산 상황에서는 보통 결과를 한 노드로 모으거나, 각 노드별 partial 결과를 별도 저장

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
