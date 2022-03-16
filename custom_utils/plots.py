import cv2
import numpy as np


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_detection(img, bboxes, scores, cls_ids, conf=0.5, class_names=None, hide_conf=False, hide_id=False,
                   line_thickness=2, font_size=1, font_thickness=1):
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        if score < conf:
            continue
        bbox_int = list(map(int, bbox))
        cls_id = int(cls_id)
        color = colors(cls_id, True)
        cv2.rectangle(img, bbox_int[:2], bbox_int[2:], color, line_thickness)
        if cls_id >= len(class_names):
            cat = cls_id
        else:
            cat = class_names[cls_id]
        if not hide_conf:
            text = f'{cat}: {score * 100:.1f}%'
        else:
            text = f'{cat}'
        if not hide_id:
            plot_label(img, bbox_int, text, color, font_size, font_thickness)
    return img


def plot_tracking(img, xywhs, obj_ids, hide_id=False, line_thickness=2, font_size=1, font_thickness=1):
    for xywh, obj_id in zip(xywhs, obj_ids):
        bbox_int = list(map(int, (xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3])))
        obj_id = int(obj_id)
        color = colors(obj_id, True)
        cv2.rectangle(img, bbox_int[:2], bbox_int[2:], color, line_thickness)
        text = f"{obj_id}"
        if not hide_id:
            plot_label(img, bbox_int, text, color, font_size, font_thickness)
    return img


def plot_label(img, xyxy, label, color, font_size=1., font_thickness=1):
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, (xyxy[0], xyxy[1] - int(txt_size[1] * 1.5)), (xyxy[0] + txt_size[0] + 1, xyxy[1]),
                  txt_bk_color, -1)
    cv2.putText(img, label, (xyxy[0], xyxy[1] - int(txt_size[1] * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)
