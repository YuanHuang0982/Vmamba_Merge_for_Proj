
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


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

    def enable_dis_for_all_ss2dv2(model, every=5, q=0.90):
        cnt = 0
        for name, m in model.named_modules():
            # SS2Dv2 계열이면 보통 이런 속성들이 있음
            if hasattr(m, "forward_corev2") and hasattr(m, "dt_projs_bias") and hasattr(m, "k_group"):
                m.extract_dis = True
                m.extract_dis_every = every
                m.extract_dis_q = q
                cnt += 1
        print("[DIS] matched modules:", cnt)

    # ---- 여기 추가 ----
    enable_dis_for_all_ss2dv2(model, every=5, q=0.90)
    # -------------------

    # ✅ 여기부터 추가 (do_inference 호출 전에)
    print("[DEBUG] TEST.RE_RANKING =", cfg.TEST.RE_RANKING)
    print("[DEBUG] TEST.DIST_MAT   =", cfg.TEST.DIST_MAT)
    if getattr(cfg.TEST, "DIST_MAT", ""):
        print("[DEBUG] DIST_MAT exists?", os.path.exists(cfg.TEST.DIST_MAT), "| path =", cfg.TEST.DIST_MAT)
    print("[DEBUG] num_query =", num_query)
    print("[DEBUG] len(val_loader.dataset) =", len(val_loader.dataset) if hasattr(val_loader, "dataset") else "NA")
    # ✅ 여기까지 추가

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
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)