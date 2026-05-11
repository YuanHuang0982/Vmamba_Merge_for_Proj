import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

import time
import numpy as np
import torch
from collections import defaultdict
from fvcore.nn import FlopCountAnalysis


def _get_test_hw_from_cfg(cfg):
    # TransReID 계열은 보통 INPUT.SIZE_TEST = [256, 128]
    if hasattr(cfg, "INPUT") and hasattr(cfg.INPUT, "SIZE_TEST"):
        h, w = cfg.INPUT.SIZE_TEST
        return int(h), int(w)
    # fallback
    return 256, 128

def _first_tensor(vals):
    for v in vals:
        if isinstance(v, torch.Tensor):
            return v
    return None

def _jit_numel(t):
    # fvcore 핸들에서 ins/outs가 torch.Tensor일 수도 있고,
    # JIT Value 형태일 수도 있어서 최대한 안전하게 처리
    try:
        if isinstance(t, torch.Tensor):
            return int(t.numel())
    except Exception:
        pass

    try:
        # JIT Value -> type().sizes() 패턴
        tt = t.type()
        sizes = tt.sizes() if hasattr(tt, "sizes") else None
        if not sizes:
            return 0
        n = 1
        for s in sizes:
            if s is None:
                return 0
            n *= int(s)
        return int(n)
    except Exception:
        return 0

def _const_int(v, default=1):
    # topk 같은 op에서 k 뽑기
    try:
        if v.node().kind() == "prim::Constant":
            iv = v.toIValue()
            if isinstance(iv, int):
                return iv
    except Exception:
        pass
    return default

def _elemwise(ins, outs):
    x = _first_tensor(ins) or (_first_tensor(outs) if outs else None)
    return _jit_numel(x)

def _reduce_op(ins, outs):
    # mean/std/sum/abs 등: 원소당 1~2 FLOP 근사
    x = _first_tensor(ins) or (_first_tensor(outs) if outs else None)
    return _jit_numel(x)

def _pythonop_approx(ins, outs, weight=10):
    x = _first_tensor(ins) or (_first_tensor(outs) if outs else None)
    return _jit_numel(x) * int(weight)

def _selectivescan_fallback(ins, outs):
    # selective_scan_flop_jit가 없을 때 fallback 근사
    # 보수적으로 크게 잡고 싶으면 weight를 키워라(예: 50)
    return _pythonop_approx(ins, outs, weight=30)

def register_op_handles_for_vssm(flops: FlopCountAnalysis, selective_scan_flop_jit=None):
    """
    네 로그에 뜬 unsupported op들을 중심으로 handle 등록.
    - elementwise / activation: numel 기반 근사
    - CrossScan/CrossMerge: PythonOp 근사
    - SelectiveScanCuda: 가능하면 selective_scan_flop_jit 사용, 아니면 fallback
    """

    # ---------- CUDA / PythonOp 계열 ----------
    # SelectiveScan
    for name in [
        "prim::PythonOp.SelectiveScanCuda",
        "PythonOp.SelectiveScanCuda",
        "prim::PythonOp.SelectiveScan",
        "PythonOp.SelectiveScan",
    ]:
        if selective_scan_flop_jit is not None:
            flops.set_op_handle(name, selective_scan_flop_jit)
        else:
            flops.set_op_handle(name, _selectivescan_fallback)

    # CrossScan/CrossMerge (Triton)
    for name in [
        "prim::PythonOp.CrossScanTritonF",
        "PythonOp.CrossScanTritonF",
        "prim::PythonOp.CrossScan",
        "PythonOp.CrossScan",
        "prim::PythonOp.CrossMergeTritonF",
        "PythonOp.CrossMergeTritonF",
        "prim::PythonOp.CrossMerge",
        "PythonOp.CrossMerge",
    ]:
        flops.set_op_handle(name, lambda ins, outs: _pythonop_approx(ins, outs, weight=10))

    # ---------- activation (근사 가중치) ----------
    # gelu/silu/softplus/sigmoid는 elementwise보다 FLOPs 조금 더 주는 게 일반적
    flops.set_op_handle("aten::gelu",     lambda ins, outs: _reduce_op(ins, outs) * 4)
    flops.set_op_handle("aten::silu",     lambda ins, outs: _reduce_op(ins, outs) * 3)
    flops.set_op_handle("aten::softplus", lambda ins, outs: _reduce_op(ins, outs) * 3)
    flops.set_op_handle("aten::sigmoid",  lambda ins, outs: _reduce_op(ins, outs) * 4)

    # ---------- elementwise ----------
    for op in [
        "aten::add",
        "aten::mul",
        "aten::sub",
        "aten::div",
        "aten::exp",
        "aten::neg",
        "aten::where",
        "aten::clamp_min",
    ]:
        flops.set_op_handle(op, _elemwise)

    # ---------- reductions / stats ----------
    flops.set_op_handle("aten::abs",  _reduce_op)
    flops.set_op_handle("aten::mean", _reduce_op)
    flops.set_op_handle("aten::sum",  _reduce_op)

    # std는 mean + sqrt 느낌이라 조금 더
    flops.set_op_handle("aten::std", lambda ins, outs: _reduce_op(ins, outs) * 2)

    # cumsum: prefix-sum이라 보통 원소당 1~2 FLOP 근사
    flops.set_op_handle("aten::cumsum", lambda ins, outs: _reduce_op(ins, outs))

    # ---------- scatter / index 계열 (근사) ----------
    # 메모리 바운드가 많지만 FLOPs만 보면 원소 수 정도로 근사
    for op in [
        "aten::scatter_add_",
        "aten::scatter_",
        "aten::index_add_",
        "aten::masked_select",
    ]:
        flops.set_op_handle(op, _reduce_op)

    # ---------- norm ----------
    # linalg_vector_norm: 제곱+합+sqrt -> 원소당 2~3 FLOP 근사
    flops.set_op_handle("aten::linalg_vector_norm", lambda ins, outs: _reduce_op(ins, outs) * 2)

    # ---------- pooling ----------
    flops.set_op_handle("aten::avg_pool2d", lambda ins, outs: _jit_numel(_first_tensor(outs)))

    # ---------- random init ----------
    # uniform_은 FLOPs보단 RNG인데, fvcore가 unsupported 찍으면 0이 되니 그냥 numel로 처리
    flops.set_op_handle("aten::uniform_", _reduce_op)

def measure_flops_fvcore(model, cfg, device="cuda", batch_size=1):
    model.eval()
    H, W = _get_test_hw_from_cfg(cfg)
    x = torch.randn(batch_size, 3, H, W, device=device)

    flops = FlopCountAnalysis(model, x)
    flops._aliases = defaultdict(lambda: "UNREGISTERED", flops._aliases)

    # 여기서 로컬로 정의
    selective_scan_flop_jit = None
    try:
        from models.csms6s import selective_scan_flop_jit as _h
        selective_scan_flop_jit = _h
    except Exception:
        try:
            from model.backbones.csms6s import selective_scan_flop_jit as _h
            selective_scan_flop_jit = _h
        except Exception:
            selective_scan_flop_jit = None

    register_op_handles_for_vssm(flops, selective_scan_flop_jit=selective_scan_flop_jit)

    total = float(flops.total())
    return total / 1e9



@torch.no_grad()
def bench_forward(model, imgs, device="cuda", warmup=50, iters=200):
    """
    imgs: (B,C,H,W) 텐서 (CPU든 GPU든 상관없음)
    return: avg_ms_per_batch, p50_ms, p90_ms, throughput_img_s, peak_mem_mb
    """
    assert device == "cuda", "CUDA에서만 torch.cuda.Event로 정확히 잰다."
    model.eval()
    torch.backends.cudnn.benchmark = True

    imgs = imgs.to(device, non_blocking=True)

    # peak memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    def _forward():
        out = model(imgs)
        # TransReID는 eval에서 (cls_score, feat) 튜플로 나오는 경우가 많음
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return out

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = _forward()
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(iters):
            starter.record()
            _ = _forward()
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms / batch

    times_sorted = sorted(times)
    avg_ms = float(sum(times) / len(times))
    p50 = float(times_sorted[int(0.50 * (len(times_sorted) - 1))])
    p90 = float(times_sorted[int(0.90 * (len(times_sorted) - 1))])

    bsz = int(imgs.shape[0])
    throughput = (bsz * 1000.0) / avg_ms  # img/s

    peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return avg_ms, p50, p90, throughput, peak_mem_mb


def batch_sweep_on_loader(model, loader, device="cuda",
                          bs_list=(1, 8, 16, 32, 64, 128),
                          warmup=50, iters=200):
    """
    loader에서 첫 배치를 꺼내서 imgs[:bs]로 sweep.
    (bs별로 DataLoader 재생성 안해도 리뷰어 설득용 표는 충분히 나옴)
    """
    batch = next(iter(loader))
    # TransReID dataloader는 보통 (img, pid, camid, viewid, ...) 형태
    imgs = batch[0]

    results = []
    for bs in bs_list:
        if bs > imgs.shape[0]:
            continue
        x = imgs[:bs].contiguous()
        avg_ms, p50, p90, thr, peak = bench_forward(model, x, device=device, warmup=warmup, iters=iters)
        results.append((bs, avg_ms, p50, p90, thr, peak))
        print(f"[SWEEP] bs={bs:>3} | avg={avg_ms:.3f} ms/batch | p50={p50:.3f} | p90={p90:.3f} | thr={thr:.2f} img/s | peak={peak:.1f} MB")

    print("\nBatch sweep summary:")
    print("bs | avg(ms/batch) | p50 | p90 | img/s | peak(MB)")
    for bs, avg_ms, p50, p90, thr, peak in results:
        print(f"{bs:>3} | {avg_ms:>12.3f} | {p50:>5.3f} | {p90:>5.3f} | {thr:>7.2f} | {peak:>8.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)


    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # ---------------------------
    # FLOPs (GFLOPs)
    # ---------------------------
    if device == "cuda":
        gflops = measure_flops_fvcore(model, cfg, device=device, batch_size=1)
        logger.info(f"[FLOPs] batch=1 기준: {gflops:.2f} GFLOPs")
    else:
        logger.info("[FLOPs] CUDA가 아니라서 생략")

    # ---------------------------
    # Latency / Throughput (Query: val_loader 첫 배치 기준)
    # ---------------------------
    if device == "cuda":
        logger.info("=== Latency/Throughput batch sweep on val_loader (first batch) ===")
        batch_sweep_on_loader(
            model, val_loader, device=device,
            bs_list=(1, 8, 16, 32, 64, 128),
            warmup=50, iters=200
        )
    else:
        logger.info("[Latency/Throughput] CUDA가 아니라서 생략")
   

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
        with torch.inference_mode():
            do_inference(cfg, model, val_loader, num_query)

