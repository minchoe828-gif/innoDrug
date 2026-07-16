import sys
import torch
import re
from pathlib import Path

def get_cell_dino_model(
    repo_dir='/workspace/innoDrug/clustering/dinov2', 
    weight_path='/workspace/innoDrug/clustering/weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth'
):
    """
    Channel-Adaptive Cell-DINO 모델을 생성하고 가중치를 입혀서 반환합니다.
    """
    repo_path = Path(repo_dir)
    if str(repo_path) not in sys.path:
        sys.path.append(str(repo_path))

    try:
        from dinov2.models.vision_transformer import vit_large
    except ImportError:
        raise ImportError(f"'{repo_dir}' 경로에서 dinov2 모듈을 찾을 수 없습니다. 경로를 확인해주세요.")

    # 1. 뼈대 생성
    # 에러 원인 1 해결: 이 가중치는 1채널(흑백) 전용이므로 in_chans=1 로 설정합니다.
    model = vit_large(
        patch_size=16, 
        img_size=224, 
        init_values=1e-5, 
        block_chunks=0,
        in_chans=1,             # 👈 3에서 1로 수정!
        channel_adaptive=True
    )

    weight_file = Path(weight_path)
    if not weight_file.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_file}")
        
    # weights_only=True 경고창을 없애기 위한 최신 파이토치 권장 옵션 추가
    state_dict = torch.load(weight_file, map_location='cpu', weights_only=True)

    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'teacher' in state_dict:
        state_dict = state_dict['teacher']

    # 2. 가중치 이름(Key) 다듬기
    new_state_dict = {}
    for k, v in state_dict.items():
        # 분산학습 module. 찌꺼기 제거
        k = k.replace("module.", "")
        
        # 에러 원인 2 해결: 블록 청크 찌꺼기 제거 
        # (예: 'blocks.1.6.norm1' -> 'blocks.6.norm1' 으로 변환)
        k = re.sub(r'blocks\.\d+\.(\d+)\.', r'blocks.\1.', k)
        
        new_state_dict[k] = v

    # 3. 가중치 적용
    # 이름과 사이즈를 완벽히 맞췄으므로 strict=True를 무사히 통과할 것입니다.
    model.load_state_dict(new_state_dict, strict=True)

    # 4. 디바이스 할당 및 평가 모드 전환
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model

# --- 테스트 실행부 ---
if __name__ == "__main__":
    print("Loading model...")
    model = get_cell_dino_model()
    current_device = next(model.parameters()).device
    print(f"✅ Model successfully loaded on {current_device}!")