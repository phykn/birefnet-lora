"""
BiRefNet 모델 설정.
원본 config.py에서 모델 구조에 필요한 설정만 남기고 나머지(데이터셋 경로, 학습 스케줄, Loss 람다 등) 제거.
"""


class Config:
    def __init__(self):
        # ── 모델 구조 설정 ──
        self.bb = 'swin_v1_l'
        self.batch_size = 8
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = 3
        self.mul_scl_ipt = 'cat'
        self.dec_att = 'ASPPDeformable'
        self.squeeze_block = 'BasicDecBlk_x1'
        self.dec_blk = 'BasicDecBlk'
        self.lat_blk = 'BasicLatBlk'
        self.dec_channels_inter = 'fixed'
        self.auxiliary_classification = False
        self.freeze_bb = False

        # 채널 설정
        self.lateral_channels_in_collection = {
            'vgg16': [512, 512, 256, 128], 'vgg16bn': [512, 512, 256, 128], 'resnet50': [2048, 1024, 512, 256],
            'swin_v1_l': [1536, 768, 384, 192], 'swin_v1_b': [1024, 512, 256, 128],
            'swin_v1_s': [768, 384, 192, 96], 'swin_v1_t': [768, 384, 192, 96],
            'pvt_v2_b5': [512, 320, 128, 64], 'pvt_v2_b2': [512, 320, 128, 64],
            'pvt_v2_b1': [512, 320, 128, 64], 'pvt_v2_b0': [256, 160, 64, 32],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [ch * 2 for ch in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        # SDPA
        self.SDPA_enabled = True
