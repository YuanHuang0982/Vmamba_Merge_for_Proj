from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = 0

# backbone
_C.MODEL.NAME = "vssm"              # make_model에서 assert로 걸어둔 값과 맞추기

# pretrained backbone loading
_C.MODEL.PRETRAIN_CHOICE = "none"   # "imagenet" or "pretrained"면 backbone ckpt 로드
_C.MODEL.PRETRAIN_PATH = ""         # 나중에 로딩 연결할 때 사용

# drop path (VMambaReID가 이 값을 사용)
_C.MODEL.DROP_PATH_RATE = 0.2

# VSSM settings
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.EMBED_DIM = 128
_C.MODEL.VSSM.DEPTHS = [2, 2, 20, 2]

_C.MODEL.VSSM.SSM_D_STATE = 1
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_RATIO = 1.0
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = False
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v05_noz"

_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.DOWNSAMPLE = "v3"     # make_model에서 downsample_version으로 전달
_C.MODEL.VSSM.PATCHEMBED = "v2"     # patchembed_version으로 전달
_C.MODEL.VSSM.NORM_LAYER = "ln2d"

# ---- (추가) 원본 vmamba config에서 쓰는 옵션들 ----
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"

_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0

_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.POSEMBED = False
_C.MODEL.VSSM.GMLP = False

_C.MODEL.VSSM.USE_CHECKPOINT = True

# defaults.py 
_C.MODEL.VSSM.DEBUG_DELTA = False
_C.MODEL.VSSM.DEBUG_EVERY = 200

# -----------------------------------------------------------------------------
# Token Merge / DIS-based Merge
# -----------------------------------------------------------------------------
_C.MODEL.VSSM.USE_TOKEN_MERGE = True

# 0-index 기준
# 0: stage1, 1: stage2, 2: stage3, 3: stage4
_C.MODEL.VSSM.MERGE_STAGE = 2

# stage3 내부 몇 번째 block 뒤에서 merge할지
# DEPTHS = [2, 2, 20, 2]이면 stage3 block index는 0~19
_C.MODEL.VSSM.MERGE_AFTER_BLOCK = [10]

# WeightedTokenMerge2D softmax temperature
_C.MODEL.VSSM.MERGE_TAU = 1.0

# score_map = alpha * (1 - DIS) + (1 - alpha) * feature_norm
_C.MODEL.VSSM.MERGE_ALPHA = 0.6

# merge 자체가 H/2, W/2로 줄이므로 보통 False 권장
_C.MODEL.VSSM.MERGE_DOWNSAMPLE = False

# neck / loss heads
_C.MODEL.NECK = "bnneck"
_C.MODEL.COS_LAYER = False
_C.MODEL.IF_WITH_CENTER = "no"

_C.MODEL.ID_LOSS_TYPE = "softmax"
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = "triplet"

_C.MODEL.DIST_TRAIN = False
_C.MODEL.NO_MARGIN = False
_C.MODEL.IF_LABELSMOOTH = "on"

_C.MODEL.CAM_SCALE = 0.1
_C.MODEL.VIEW_SCALE = 0.1

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# ReID 표준 비율 권장
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TEST  = [256, 128]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD  = [0.229, 0.224, 0.225]
_C.INPUT.PADDING = 10

_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.0
# timm auto_augment string 예: 'rand-m9-mstd0.5-inc1', 'original', 'v0', 'none'
_C.AUG.AUTO_AUGMENT = 'none'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = "market1501"
_C.DATASETS.ROOT_DIR = "../data"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = "softmax"
_C.DATALOADER.NUM_INSTANCE = 16

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# optimizer 그룹(decay/no_decay) 기반이면 AdamW가 자연스러움
_C.SOLVER.OPTIMIZER_NAME = "AdamW"

_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.SEED = 1234
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# weight decay: VMamba/ViT류는 보통 ResNet보다 크게 주는 편
# 일단 보수적으로 0.01부터 시작 추천 (0.0005는 너무 약할 가능성 큼)
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0   # optimizer에서 bias/norm은 no_decay로 빼므로 의미 거의 없음

# scheduler
_C.SOLVER.WARMUP_EPOCHS = 5

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 10
_C.SOLVER.IMS_PER_BATCH = 64

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = "after"
_C.TEST.FEAT_NORM = "yes"
_C.TEST.DIST_MAT = "dist_mat.npy"
_C.TEST.EVAL = False

# --- Visualization (retrieval strip) ---
_C.TEST.VISUALIZE = False          # True면 시각화 저장
_C.TEST.VIS_DIR = ""               # 비우면 OUTPUT_DIR/vis 사용
_C.TEST.VIS_TOPK = 10              # query당 top-k gallery 저장

_C.TEST.VIS_MAP = False
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""


