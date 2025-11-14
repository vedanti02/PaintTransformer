import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import network
import os
import math
import render_utils
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import cv2
import render_parallel
import glob

def process_one_image(img_file, net_g, meta_brushes, output_dir):
    """Process a single image and save output + strokes."""

    img_name = os.path.splitext(os.path.basename(img_file))[0]
    out_path = os.path.join(output_dir, f"{img_name}.png")
    stroke_path = os.path.join("strokes", f"{img_name}.pt")

    print(f"\n[INFO] Processing {img_file}")

    # Load input
    original_img = render_utils.read_img(img_file, "RGB", 512, 512)

    # Run model
    final_result, stroke_buffer = render_parallel.render_parallel(
        original_img, net_g, meta_brushes
    )

    # Save painting
    cv2.imwrite(out_path, final_result)
    print(f"[INFO] Saved output → {out_path}")

    # Save strokes
    paddle.save(stroke_buffer, stroke_path)
    print(f"[INFO] Saved strokes → {stroke_path}")

def main(input_path, model_path, output_dir, need_animation=False, resize_h=None, resize_w=None, serial=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, input_name)
    stroke_num = 8

    #* ----- load model ----- *#
    paddle.set_device('gpu')
    net_g = network.Painter(5, stroke_num, 256, 8, 3, 3)
    net_g.set_state_dict(paddle.load(model_path))
    net_g.eval()
    for param in net_g.parameters():
        param.stop_gradient = True

    #* ----- load brush ----- *#
    brush_large_vertical = render_utils.read_img('brush/brush_large_vertical.png', 'L')
    brush_large_horizontal = render_utils.read_img('brush/brush_large_horizontal.png', 'L')
    meta_brushes = paddle.concat([brush_large_vertical, brush_large_horizontal], axis=0)

    # -------------------------
    # Determine images
    # -------------------------
    if os.path.isdir(input_path):
        image_list = sorted(
            glob.glob(os.path.join(input_path, "*.jpg")) +
            glob.glob(os.path.join(input_path, "*.png")) +
            glob.glob(os.path.join(input_path, "*.jpeg"))
        )
    else:
        image_list = [input_path]

    print(f"[INFO] Found {len(image_list)} images")

    import time
    t0 = time.time()
    
    for img_file in image_list:
        process_one_image(img_file, net_g, meta_brushes, output_dir)
    
    print("total infer time:", time.time() - t0)

if __name__ == '__main__':
    
    main(input_path='celebA/',
         model_path='paint_best.pdparams',
         output_dir='output/',
         need_animation=False,  # whether need intermediate results for animation.
         resize_h=512,         # resize original input to this size. None means do not resize.
         resize_w=512,         # resize original input to this size. None means do not resize.
         serial=False)          # if need animation, serial must be True.