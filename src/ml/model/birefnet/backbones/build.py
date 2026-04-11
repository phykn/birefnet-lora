from .swin_v1 import SwinTransformer, swin_v1_l


def build_backbone(gradient_checkpointing: bool = False) -> SwinTransformer:
    return swin_v1_l(gradient_checkpointing=gradient_checkpointing)
