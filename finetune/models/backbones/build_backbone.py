"""Backbone 빌더. swin_v1 전용."""

import os
import torch
from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l


def build_backbone(bb_name, pretrained=True, params_settings=''):
    bb = eval('{}({})'.format(bb_name, params_settings))
    if pretrained:
        bb = load_weights(bb, bb_name)
    return bb


# 가중치 파일 검색 경로 (코드 수정 없이 여기만 추가하면 됨)
WEIGHT_DIRS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'weights'),   # finetune/weights/
    os.path.expanduser('~/.cache/birefnet'),
]

WEIGHT_FILENAMES = {
    'swin_v1_l': 'swin_large_patch4_window12_384_22kto1k.pth',
    'swin_v1_b': 'swin_base_patch4_window12_384_22kto1k.pth',
    'swin_v1_s': 'swin_small_patch4_window7_224_22kto1k.pth',
    'swin_v1_t': 'swin_tiny_patch4_window7_224_22kto1k.pth',
}


def load_weights(model, model_name):
    """로컬 파일에서 가중치 로드."""
    filename = WEIGHT_FILENAMES.get(model_name)
    if not filename:
        print(f'No weight filename defined for {model_name}')
        return model

    # 검색 경로에서 파일 찾기
    for d in WEIGHT_DIRS:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            print(f'Loading weights: {path}')
            save_model = torch.load(path, map_location='cpu', weights_only=True)
            break
    else:
        print(f'Weight file not found: {filename}')
        print(f'Place it in one of: {[os.path.abspath(d) for d in WEIGHT_DIRS]}')
        return model

    # state_dict 매핑
    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in save_model.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    if not state_dict:
        for key in save_model:
            if isinstance(save_model[key], dict):
                state_dict = {
                    k: v for k, v in save_model[key].items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                if state_dict:
                    break
    if not state_dict:
        print('Warning: No matching weights found in file.')
        return model

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print(f'Loaded {len(state_dict)}/{len(model_dict)} weight tensors.')
    return model
