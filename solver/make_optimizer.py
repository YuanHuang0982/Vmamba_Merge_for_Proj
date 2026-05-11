import torch
import torch.nn as nn

def make_optimizer(cfg, model, center_criterion):
    base_lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY

    # --- (기존) no_weight_decay / keywords 수집 ---
    nwd_names = set()
    nwd_keywords = set()
    for obj in [model, getattr(model, "base", None)]:
        if obj is None:
            continue
        if hasattr(obj, "no_weight_decay") and callable(getattr(obj, "no_weight_decay")):
            try:
                nwd_names |= set(obj.no_weight_decay())
            except Exception:
                pass
        if hasattr(obj, "no_weight_decay_keywords") and callable(getattr(obj, "no_weight_decay_keywords")):
            try:
                nwd_keywords |= set(obj.no_weight_decay_keywords())
            except Exception:
                pass

    def is_no_weight_decay(name: str) -> bool:
        if name in nwd_names:
            return True
        for kw in nwd_keywords:
            if kw and (kw in name):
                return True
        if name.endswith(".bias"):
            return True
        if any(x in name.lower() for x in ["norm", "bn", "ln", "layernorm", "batchnorm"]):
            return True
        return False

    def is_fc(name: str) -> bool:
        return ("classifier" in name) or ("arcface" in name) or ("cosface" in name) or ("amsoftmax" in name) or ("circle" in name)

    # -----------------------------
    # 2) 파라미터 그룹 구성 (decay / no_decay + delta 전용)
    # -----------------------------
    decay_params = []
    nodecay_params = []
    decay_params_lr = []
    nodecay_params_lr = []

    # ✅ delta 전용 버킷 추가
    delta_params = []
    delta_params_lr = []  # 혹시 LARGE_FC_LR 경로로 들어가면 분리(보통은 비어있을 가능성 큼)

    # delta param 이름 규칙
    DELTA_KEYS = ["tau_raw", "gamma_raw", "smin_raw", "srange_raw"]
    def is_delta_param(name: str) -> bool:
        return any(k in name for k in DELTA_KEYS)

    # delta lr 비율 (고정 0.1로 쓰고 싶으면 여기 그대로)
    delta_lr_scale = 0.1

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # lr 설정(기존 로직 유지)
        lr = base_lr
        if cfg.SOLVER.LARGE_FC_LR and is_fc(name):
            lr = base_lr * 2

        # ✅ 1) delta param은 최우선으로 빼서 별도 그룹으로
        if is_delta_param(name):
            if lr != base_lr:
                delta_params_lr.append(p)
            else:
                delta_params.append(p)
            continue  # <- 중요: 아래 decay/no_decay로 안 내려가게

        # ✅ 2) 나머지는 기존 decay/no_decay 로직
        if is_no_weight_decay(name):
            if lr != base_lr:
                nodecay_params_lr.append(p)
            else:
                nodecay_params.append(p)
        else:
            if lr != base_lr:
                decay_params_lr.append(p)
            else:
                decay_params.append(p)

    param_groups = []
    if len(decay_params) > 0:
        param_groups.append({"params": decay_params, "lr": base_lr, "weight_decay": wd})
    if len(nodecay_params) > 0:
        param_groups.append({"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0})
    if len(decay_params_lr) > 0:
        param_groups.append({"params": decay_params_lr, "lr": base_lr * 2, "weight_decay": wd})
    if len(nodecay_params_lr) > 0:
        param_groups.append({"params": nodecay_params_lr, "lr": base_lr * 2, "weight_decay": 0.0})

    # ✅ 3) delta 전용 그룹을 마지막에 추가 (wd=0, lr은 더 작게)
    if len(delta_params) > 0:
        param_groups.append({"params": delta_params, "lr": base_lr * delta_lr_scale, "weight_decay": 0.0})
    if len(delta_params_lr) > 0:
        param_groups.append({"params": delta_params_lr, "lr": base_lr * 2 * delta_lr_scale, "weight_decay": 0.0})

    # -----------------------------
    # 3) Optimizer 생성 (기존 그대로)
    # -----------------------------
    opt_name = cfg.SOLVER.OPTIMIZER_NAME
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.SOLVER.MOMENTUM)
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=0.0)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
    else:
        optimizer = getattr(torch.optim, opt_name)(param_groups)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center
