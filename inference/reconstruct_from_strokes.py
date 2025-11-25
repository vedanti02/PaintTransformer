# reconstruct_from_strokes.py
import os
import paddle
import paddle.nn.functional as F
import render_parallel
import render_utils
import numpy as np
import cv2

def reconstruct_from_strokes(stroke_file, output_path, brush_folder="brush"):
    print(f"[INFO] Loading stroke parameters: {stroke_file}")
    stroke_buffer = paddle.load(stroke_file)


    brush_vertical = render_utils.read_img(os.path.join(brush_folder, 'brush_large_vertical.png'), 'L')
    brush_horizontal = render_utils.read_img(os.path.join(brush_folder, 'brush_large_horizontal.png'), 'L')
    meta_brushes = paddle.concat([brush_vertical, brush_horizontal], axis=0)

    # ---------------- DETERMINE CANVAS SIZE ----------------
    # Use the largest layer's patch count to determine final canvas size
    final_layer = stroke_buffer[-1]
    _, patch_h, patch_w, stroke_num, param_dim = final_layer.shape
    # Each patch size in original PaintTransformer is 32, adjust canvas
    patch_size = 32
    canvas_h = patch_h * patch_size
    canvas_w = patch_w * patch_size

    #blank canvas
    canvas = paddle.zeros([1, 3, canvas_h, canvas_w], dtype='float32')

    stroke_buffer =stroke_buffer[:5]

    for layer_idx, layer_param in enumerate(stroke_buffer):
        print(f"Rendering layer {layer_idx+1}/{len(stroke_buffer)}  shape={layer_param.shape}")
        canvas = render_parallel.param2img_parallel(layer_param, meta_brushes, canvas, stroke_num=layer_param.shape[3])

#save
    canvas_np = (canvas.numpy().squeeze().transpose([1,2,0])[:,:,::-1] * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas_np)
    print(f"[INFO] Saved reconstructed image: {output_path}")
    return canvas_np

if __name__ == "__main__":
    reconstruct_from_strokes(
        stroke_file="/content/drive/MyDrive/PaintTransformer/strokes/000001.pt",
        output_path="reconstructed/000001.png"
    )
