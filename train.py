from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
import config
import re


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    print("CONFIG FILE:", config.__file__)
    print("BEFORE MERGE DEVICE_ID:", cfg.MODEL.DEVICE_ID, type(cfg.MODEL.DEVICE_ID))
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)


    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    #print("[DEBUG] num_query =", num_query)
    #print("[DEBUG] expected MSMT17 query =", 11659)  # 참고용(로그에 나온 값)

    def parse_cam_from_path_msmt(p):
        # MSMT17 파일명 예: 0000_031_14_0303morning_0245_0.jpg  -> cam=14 (1-based)
        bn = os.path.basename(p)
        parts = bn.split('_')
        if len(parts) >= 3 and parts[2].isdigit():
            return int(parts[2])
        return None

    def unpack_batch_for_debug(batch):
        # val_loader가 (img, pid, camid, camids, view, imgpath) 형태인 경우를 우선 지원
        if isinstance(batch, (list, tuple)):
            L = len(batch)
            if L >= 6:
                return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            if L == 5:
                return batch[0], batch[1], batch[2], batch[3], batch[4], None
            if L == 4:
                return batch[0], batch[1], batch[2], batch[2], batch[3], None
        raise ValueError(f"Unknown batch format: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

    # val 첫 배치에서만 camid vs path_cam 확인
    batch = next(iter(val_loader))
    img, pid, camid, camids, view, imgpath = unpack_batch_for_debug(batch)

    camid10 = [int(x) for x in list(camid)[:10]] if not torch.is_tensor(camid) else camid[:10].detach().cpu().tolist()

    print("\n=== [DEBUG] val first-batch cam check ===")
    print("camid[:10]   =", camid10)

    if imgpath is None:
        print("[DEBUG] imgpath is None -> val_loader가 path를 안 주는 로더임. datasets/make_dataloader 쪽에서 imgpath를 반환하도록 확인 필요.")
    else:
        paths10 = list(imgpath)[:10]
        parsed = [parse_cam_from_path_msmt(p) for p in paths10]
        print("path_cam[:10]=", parsed)
        print("path_cam-1   =", [x-1 if x is not None else None for x in parsed])
        print("basename     =", [os.path.basename(p) for p in paths10])


    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
