import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from YOLOx.yolox.exp import get_exp as get_yolox_exp
from YOLOx.yolox.utils import fuse_model, postprocess
from BYTE.byte_tracker import BYTETracker

warnings.filterwarnings("ignore")
FILE = Path(__file__).absolute()

if os.path.join(FILE.parents[0], "custom_utils") not in sys.path:
    sys.path.append(os.path.join(FILE.parents[0], "custom_utils"))
from custom_utils.general import (select_device, increment_path, model_info, preprocess, scale_coords,
                                  filtering_targets, xyxy2cpwhn)
from custom_utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from custom_utils.plots import plot_detection, plot_tracking


@torch.no_grad()
def main(args):
    # Load arguments
    source = args.source
    out_dir = args.out_dir
    run_name = args.run_name
    save_vid = args.save_vid
    save_label = args.save_label
    half = args.half
    fuse = args.fuse
    device = args.device
    is_video_frames = args.is_video_frames
    view_mode = args.view_mode
    view_conf_thr = args.view_conf_thr
    hide_label = args.hide_label
    hide_conf = args.hide_conf
    hide_id = args.hide_id
    view = args.view

    yolo_imgsz = args.yolo_imgsz
    yolox_exp = args.yolox_exp
    yolox_name = args.yolox_name
    yolox_weights = args.yolox_weights
    yolox_conf_thr = args.yolox_conf_thr
    yolox_iou_thr = args.yolox_iou_thr

    min_box_area = args.min_box_area

    # Set device(cpu/gpu) and make directory for save results
    device = select_device(device)
    save_dir = increment_path(Path(out_dir) / run_name, exist_ok=False)
    if save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_label:
            label_save_dir = os.path.join(save_dir, "labels")
            os.mkdir(label_save_dir)

    # Load YOLOx (person detector)
    yolox_exp = get_yolox_exp(yolox_exp, yolox_name)
    yolox_model = yolox_exp.get_model().to(device)
    yolox_model.eval()
    if os.path.isfile(yolox_weights):
        ckpt = torch.load(yolox_weights)
        yolox_model.load_state_dict(ckpt["model"])
    if fuse:
        yolox_model = fuse_model(yolox_model)
    if half:
        yolox_model.half()
    model_info(yolox_model, model_name="YOLOX")

    # Make dataloader for inference
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=yolo_imgsz)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Initialize BYTE (person tracker)
    trackers = [BYTETracker(args) for _ in range(bs)]

    # Run inference
    if device.type != "cpu":
        yolox_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolox_model.parameters())))
    for path, im, im0s, vid_cap, s, resize_params in dataset:
        print("\n---")
        print(im.shape)
        ts = time.time()
        # Image preprocessing (scaling + normalize)
        im = preprocess(im, device, half)

        # YOLOX prediction
        t1 = time.time()
        yolox_preds = yolox_model(im)
        yolox_preds = postprocess(yolox_preds, yolox_exp.num_classes, yolox_conf_thr, yolox_iou_thr)
        t2 = time.time()
        print(f"\tyolox prediction time: {t2 - t1:.4f}")

        for i, yolox_pred in enumerate(yolox_preds):
            if webcam:
                p, im0, imv, cap, resize_param = path[i], im0s[i].copy(), im0s[i].copy(), vid_cap[i], resize_params[i]
            else:
                p, im0, imv, cap, resize_param = path, im0s.copy(), im0s.copy(), vid_cap, resize_params
            p = Path(p)
            save_path = str(save_dir / "video") if is_video_frames and dataset.mode == "image" \
                else str(save_dir / p.name)

            if yolox_pred is not None:
                yolox_pred[:, :4] = scale_coords(im.shape[2:], yolox_pred[:, :4], im0.shape[:2])
                if save_label:
                    txt = ""
                    pred_for_label = xyxy2cpwhn(yolox_pred, w=imv.shape[1], h=imv.shape[0])
                    for pred in pred_for_label:
                        conf = pred[4] * pred[5]
                        if conf >= view_conf_thr:
                            txt += f"{pred[-1]} {pred[0]:.6f} {pred[1]:.6f} {pred[2]:.6f} {pred[3]:.6f}\n"
                    if txt != "":
                        label_save_name = ".".join(p.name.split(".")[:-1]) + ".txt"
                        label_save_path = os.path.join(label_save_dir, label_save_name)
                        with open(label_save_path, "w") as f:
                            f.write(txt)

                # BYTE prediction (tracking)
                t1 = time.time()
                online_targets = trackers[i].update(yolox_pred, im0.shape[:2], im0.shape[:2])
                online_tlwhs, online_ids = filtering_targets(online_targets, min_box_area)
                t2 = time.time()
                print(f"\ttracker prediction time: {t2 - t1:.4f}")

                t1 = time.time()
                if not hide_label:
                    if view_mode == 1:
                        if yolox_pred is not None:
                            bboxes = yolox_pred[:, :4]
                            scores = yolox_pred[:, 4] * yolox_pred[:, 5]
                            clss = yolox_pred[:, 6]
                            imv = plot_detection(imv, bboxes, scores, clss, view_conf_thr, ["person"],
                                                 hide_conf, hide_id)
                    elif view_mode == 2:
                        imv = plot_tracking(imv, online_tlwhs, online_ids,
                                            hide_id=hide_id)
                t2 = time.time()
                print(f"\tvisualization time: {t2 - t1:.4f}")
            else:
                online_tlwhs = []
                online_ids = []

            # Visualize prediction results
            if view:
                img_name = f"{Path(save_path).name}_{i}"
                cv2.imshow(img_name, imv)
                cv2.waitKey(1)

            # Save results
            if save_vid:
                if dataset.mode == "image" and not is_video_frames:
                    cv2.imwrite(save_path, imv)
                elif dataset.mode == "image" and is_video_frames:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        fps, w, h = 30, imv.shape[1], imv.shape[0]
                        save_path += ".mp4"
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(imv)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if cap is not None:
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, imv.shape[1], imv.shape[1]
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(imv)
        tf = time.time()
        print(f"total: {tf - ts:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments for loading KISA dataset
    source = "0"
    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "rtsp://datonai:datonai@172.30.1.50:554/stream1"
    #source = "rtsp://datonai:datonai@172.30.1.24:554/stream1"
    source = "rtsp://datonai:datonai@172.30.1.1:554/stream1"
    source = "https://www.youtube.com/watch?v=UoYchhSqPmY"
    source = "https://www.youtube.com/watch?v=uMnGzVPUEB4&t=2919s"
    #source = "https://www.youtube.com/watch?v=GKR97D4zqm8"
    #source = "/home/daton/Desktop/gs/loitering_gs/aihub_subway_cctv_1.mp4"
    #source = "source.txt"
    #source = "/media/daton/SAMSUNG/4. 민간분야(2021 특수환경)/distribution/C032100_001.mp4"
    #source = "/media/daton/Data/datasets/MOT17/train/MOT17-04-SDP/img1"
    source = "/media/daton/Data/datasets/unlabeled_images"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--out-dir", type=str, default=f"{FILE.parents[0]}/runs/inference")
    parser.add_argument("--run-name", type=str, default="exp")
    parser.add_argument("--half", default=True, action="store_true")
    parser.add_argument("--fuse", default=True, action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--is-video-frames", action="store_true", default=False)
    parser.add_argument("--view-mode", type=int, default=1)  # 1: detection, 2: tracking
    parser.add_argument("--view-conf-thr", type=float, default=0.3)
    parser.add_argument("--hide-label", action="store_true", default=False)
    parser.add_argument("--hide-conf", action="store_true", default=False)
    parser.add_argument("--hide-id", action="store_true", default=False)
    parser.add_argument("--view", action="store_true", default=False)
    parser.add_argument("--save-vid", action="store_true", default=True)
    parser.add_argument("--save-label", action="store_true", default=True)

    # Arguments for YOLOX (person detector)
    yolox_exp = f"{FILE.parents[0]}/YOLOx/exps/example/mot/yolox_l_mix_det.py"
    #yolox_exp = f"{FILE.parents[0]}/YOLOx/exps/default/yolox_l.py"
    yolox_name = None
    yolox_weights = f"{FILE.parents[0]}/weights/yolox/bytetrack_l_mot17.pth.tar"
    #yolox_weights = f"{FILE.parents[0]}/weights/yolox/yolox_l.pth"
    parser.add_argument("--yolox-exp", type=str, default=yolox_exp)
    parser.add_argument("--yolox-name", type=str, default=yolox_name)
    parser.add_argument("--yolox-weights", type=str, default=yolox_weights)
    parser.add_argument("--yolo-imgsz", type=int, default=[1280])
    parser.add_argument("--yolox-conf-thr", type=float, default=0.01)
    parser.add_argument("--yolox-iou-thr", type=float, default=0.7)

    # Arguments for Byte (person tracker)
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=1.6)
    parser.add_argument("--min-box-area", type=float, default=10)
    parser.add_argument("--mot20", default=False, action="store_true")

    args = parser.parse_args()
    args.yolo_imgsz *= 2 if len(args.yolo_imgsz) == 1 else 1
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
