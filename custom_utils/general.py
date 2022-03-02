import os
import glob
import re
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
from thop import profile


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def select_device(device='', batch_size=None, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    if not newline:
        s = s.rstrip()
    return torch.device('cuda:0' if cuda else 'cpu')


def model_info(model, img_size=640, model_name=None, verbose=False):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    stride = 32
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device).type_as(next(model.parameters()))
    flops = profile(deepcopy(model), inputs=(img, ), verbose=False)[0]
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
    flops *= img_size[0] / stride * img_size[1] / stride * 2 / 1e9
    n_p /= 1e6
    n_g /= 1e6
    model_name = model_name if model_name is not None else "Model"
    print(f"{model_name} Summary: {len(list(model.modules()))} layers," +
          f"{n_p:.1f}M parameters, {n_g:.1f}M gradients, {flops:.2f} GFLOPs")


def preprocess(img: np.ndarray, device: torch.device, half: bool=False,
               mean: [float]=[0.485, 0.456, 0.406], std: [float]=[0.229, 0.224, 0.225], scaling=True, normalize=True):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    # Scaling pixel value 0 ~ 255 to 0 ~ 1
    if scaling:
        img /= 255.
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    # Normalize pixel value
    if normalize:
        img -= torch.Tensor(mean).reshape(3, 1, 1).to(device)
        img /= torch.Tensor(std).reshape(3, 1, 1).to(device)
    return img


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def filtering_targets(targets, min_box_area):
    online_tlwhs = []
    online_ids = []
    for t in targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > min_box_area:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
    return online_tlwhs, online_ids
