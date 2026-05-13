import torch
import torch.nn as nn
import copy

from .backbones.vmamba import VSSM

from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if getattr(m, "affine", False):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class VMambaReID(nn.Module):
    """
    TransReID 스타일 인터페이스를 유지한 VMamba(VSSM) ReID 모델.
    - train: return (cls_score, global_feat)
    - eval : return feat or global_feat (cfg.TEST.NECK_FEAT에 따라)
    """
    def __init__(self, num_classes, cfg, camera_num=0, view_num=0):
        super().__init__()

        # ---- TransReID 쪽 설정 재사용 ----
        self.neck = cfg.MODEL.NECK                 # 'no' or 'bnneck'
        self.neck_feat = cfg.TEST.NECK_FEAT        # 'after' or 'before'
        self.cos_layer = cfg.MODEL.COS_LAYER       # bool
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        # ---- VMamba/VSSM 설정 ----
        vcfg = cfg.MODEL.VSSM

        # ---- debug ----
        debug_delta = bool(getattr(vcfg, "DEBUG_DELTA", False))
        debug_every = int(getattr(vcfg, "DEBUG_EVERY", 200))

        self.base = VSSM(
            patch_size=getattr(vcfg, "PATCH_SIZE", 4),
            in_chans=getattr(vcfg, "IN_CHANS", 3),
            num_classes=0,  # ReID에서는 보통 backbone head 사용 안 함
            depths=vcfg.DEPTHS,
            dims=[vcfg.EMBED_DIM, vcfg.EMBED_DIM*2, vcfg.EMBED_DIM*4, vcfg.EMBED_DIM*8]
                 if not hasattr(vcfg, "DIMS") else vcfg.DIMS,
            ssm_d_state=vcfg.SSM_D_STATE,
            ssm_dt_rank=vcfg.SSM_DT_RANK,
            ssm_ratio=vcfg.SSM_RATIO,
            ssm_conv=vcfg.SSM_CONV,
            ssm_conv_bias=vcfg.SSM_CONV_BIAS,
            forward_type=vcfg.SSM_FORWARDTYPE,
            mlp_ratio=vcfg.MLP_RATIO,
            drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
            norm_layer=vcfg.NORM_LAYER,
            downsample_version=vcfg.DOWNSAMPLE,      
            patchembed_version=vcfg.PATCHEMBED,      
            imgsize=cfg.INPUT.SIZE_TRAIN[0] if isinstance(cfg.INPUT.SIZE_TRAIN, (list, tuple)) else cfg.INPUT.SIZE_TRAIN,

            debug_delta=debug_delta,
            debug_every=debug_every,
            use_token_merge=getattr(vcfg, "USE_TOKEN_MERGE", False),
            merge_stage=getattr(vcfg, "MERGE_STAGE", 2),
            merge_after_block=getattr(vcfg, "MERGE_AFTER_BLOCK", None),
            merge_tau=getattr(vcfg, "MERGE_TAU", 1.0),
            merge_alpha=getattr(vcfg, "MERGE_ALPHA", 0.6),
            merge_downsample=getattr(vcfg, "MERGE_DOWNSAMPLE", False),
        )


        self.channel_first = self.base.channel_first

        # patch_embed output channel (보통 dims[0] == EMBED_DIM)
        self.embed_dim0 = self.base.dims[0] if hasattr(self.base, "dims") else vcfg.EMBED_DIM

        self.in_planes = self.base.num_features
        self.num_classes = num_classes


        # camera/view num 확정
        self.camera_num = int(camera_num) if camera_num is not None else 0
        self.view_num = int(view_num) if view_num is not None else 0

        # cfg에서 스케일 읽기 (자동화용)
        cam_scale  = float(getattr(cfg.MODEL, "CAM_SCALE", 0.1))
        view_scale = float(getattr(cfg.MODEL, "VIEW_SCALE", 0.1))

        # scale=0이면 embedding 자체를 끄는 옵션 (권장)
        use_cam  = (self.camera_num > 0) and (cam_scale != 0.0)
        use_view = (self.view_num > 0)   and (view_scale != 0.0)

        # embedding
        self.cam_embed  = nn.Embedding(self.camera_num, self.embed_dim0) if use_cam else None
        self.view_embed = nn.Embedding(self.view_num,   self.embed_dim0) if use_view else None

        if self.cam_embed is not None:
            nn.init.normal_(self.cam_embed.weight, std=0.02)
        if self.view_embed is not None:
            nn.init.normal_(self.view_embed.weight, std=0.02)

        # scale은 buffer로 고정 (학습 안 함)
        if self.cam_embed is not None:
            self.register_buffer("cam_scale", torch.tensor(cam_scale, dtype=torch.float32))
        else:
            self.cam_scale = None

        if self.view_embed is not None:
            self.register_buffer("view_scale", torch.tensor(view_scale, dtype=torch.float32))
        else:
            self.view_scale = None
        

        # ---- BNNeck ----
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # ---- (추가) pretrained 로딩 ----
        if cfg.MODEL.PRETRAIN_CHOICE in ("imagenet", "pretrained") and cfg.MODEL.PRETRAIN_PATH:
            self._load_backbone_pretrained(cfg.MODEL.PRETRAIN_PATH)


        # ---- Classifier / Metric head ----
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        # (선택) imagenet pretrained 로딩을 네 VSSM에 맞게 추가
        # if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        #     self.base.load_param(cfg.MODEL.PRETRAIN_PATH)


    def load_param(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]

        new = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new[k] = v

        incompatible = self.load_state_dict(new, strict=False)
        print(f"Load full model from: {path}")
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)

    def _check_index_range(self, idx, size, name: str):
        # idx: (B,) int tensor
        if idx is None:
            return
        if idx.numel() == 0:
            return
        mn = int(idx.min().item())
        mx = int(idx.max().item())
        if mn < 0 or mx >= size:
            raise RuntimeError(
                f"[{name}] index out of range: min={mn}, max={mx}, allowed=[0, {size-1}]"
            )

    def forward_features(self, x, cam_label=None, view_label=None):
        x = self.base.patch_embed(x)

        # 레이아웃/채널 sanity check (디버깅용)
        if self.base.channel_first:
            assert x.shape[1] == self.embed_dim0, f"patch_embed out mismatch: {x.shape} vs C0={self.embed_dim0}"
        else:
            assert x.shape[-1] == self.embed_dim0, f"patch_embed out mismatch: {x.shape} vs C0={self.embed_dim0}"


        cam_s  = self.cam_scale   # tensor scalar
        view_s = self.view_scale


        # ---- cam/view embedding add (patch_embed 직후) ----
        if self.cam_embed is not None and cam_label is not None and cam_s is not None:
            cam_label = cam_label.to(x.device).long()
            self._check_index_range(cam_label, self.camera_num, "cam_label")  # ✅ 추가
            ce = self.cam_embed(cam_label)  # (B,C0)
            if self.base.channel_first:
                x = x + cam_s * ce.view(x.size(0), -1, 1, 1)
            else:
                x = x + cam_s * ce.view(x.size(0), 1, 1, -1)

        if self.view_embed is not None and view_label is not None and view_s is not None:
            view_label = view_label.to(x.device).long()
            self._check_index_range(view_label, self.view_num, "view_label")  # ✅ 추가
            ve = self.view_embed(view_label)  # (B,C0)
            if self.base.channel_first:
                x = x + view_s * ve.view(x.size(0), -1, 1, 1)
            else:
                x = x + view_s * ve.view(x.size(0), 1, 1, -1)

        # -----------------------------------------------


        if self.base.pos_embed is not None:
            pos = self.base.pos_embed
            pos = pos.permute(0, 2, 3, 1) if (not self.base.channel_first) else pos
            x = x + pos
        for layer in self.base.layers:
            if isinstance(x, tuple):
                x = x[1]  # y만 다음 stage로 전달

            out = layer(x)

            if isinstance(out, tuple):
                x = out[1]   # y (downsample 결과)
            else:
                x = out

        return x


    def pool(self, feat_map):
        if feat_map.dim() != 4:
            raise RuntimeError(f"Expected 4D feat map, got {feat_map.shape}")
        if not self.channel_first:
            feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
        global_feat = nn.functional.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # (B,C)
        return global_feat

    def forward(self, x, label=None, cam_label=None, view_label=None):
        feat_map = self.forward_features(x, cam_label=cam_label, view_label=view_label)
        global_feat = self.pool(feat_map)

        feat = self.bottleneck(global_feat) if (self.neck == 'bnneck') else global_feat

        if self.training:
            if label is None:
                raise RuntimeError("label is None in training mode")

            label = label.to(feat.device).long()

            # ✅ pid 범위 체크
            mn = int(label.min().item())
            mx = int(label.max().item())
            if mn < 0 or mx >= self.num_classes:
                raise RuntimeError(
                    f"[pid label] out of range: min={mn}, max={mx}, num_classes={self.num_classes}"
                )

            cls_score = self.classifier(feat, label) if self.ID_LOSS_TYPE in ('arcface','cosface','amsoftmax','circle') else self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat if (self.neck_feat == 'after') else global_feat

    def _load_backbone_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]

        # 1) DDP prefix 제거
        new = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new[k] = v

        # 2) 분류 head 제거
        drop_keys = [
            "classifier.head.weight",
            "classifier.head.bias",
            "head.weight",
            "head.bias",
        ]
        for k in drop_keys:
            if k in new:
                new.pop(k)

        for k in list(new.keys()):
            if k.startswith("classifier.head.") or k.startswith("head."):
                new.pop(k)

        # 3) stage3 merge 구조용 key remap
        
        #split_idx = len(self.base.layers[2].blocks_pre)
        remapped = {}

        # for k, v in new.items():
        #     if k.startswith("layers.2.blocks."):
        #         rest = k[len("layers.2.blocks."):]   # 예: "7.op.in_proj.weight"
        #         blk_str, suffix = rest.split(".", 1)
        #         blk_idx = int(blk_str)

        #         if blk_idx < split_idx:
        #             new_k = f"layers.2.blocks_pre.{blk_idx}.{suffix}"
        #         else:
        #             new_k = f"layers.2.blocks_post.{blk_idx - split_idx}.{suffix}"

        #         remapped[new_k] = v
        #     else:
        #         remapped[k] = v

        #new = remapped

        # 디버그 확인용
        print("Remapped example keys:")
        for k in list(new.keys()):
            if "layers.2.blocks_pre" in k or "layers.2.blocks_post" in k:
                print(" ", k)
                break

        incompatible = self.base.load_state_dict(new, strict=False)
        print("Load backbone ckpt:", path)
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)


def make_model(cfg, num_class, camera_num, view_num):
    model = VMambaReID(num_class, cfg, camera_num=camera_num, view_num=view_num)
    print('===========building VMamba(VSSM) ReID===========')
    return model
