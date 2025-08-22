from multiprocessing import cpu_count
from multiprocessing import Pool
import time
import os
import mxnet as mx
from mxnet import image
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional


def prepare_context(gpu_id: int = 0) -> mx.Context:
    """准备 MXNet 上下文，优先 GPU（传入 GPU id）。

    如果机器没有 GPU 或者 MXNet 未编译 GPU，可能抛错。
    """
    ctx = mx.gpu(gpu_id) if mx.context.num_gpus() > 0 else mx.cpu()
    print(f"Using context: {ctx}")
    return ctx


def load_model(ctx: mx.Context, model_name: str = 'psp_resnet101_ade'):
    """加载 GluonCV 模型（预训练）。"""
    print(f"Loading model {model_name}...")
    model = gluoncv.model_zoo.get_model(model_name, ctx=ctx, pretrained=True)
    print("Model loaded")
    return model


def load_ade_class_map(filepath: str = 'ADE20K_Class.csv') -> Dict[int, str]:
    """从 ADE20K 类别文件读取 id->name 的映射。

    期望 CSV/文本格式为每行: id,name
    """
    mapping: Dict[int, str] = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ADE class file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            try:
                idx = int(parts[0])
                name = parts[1] if len(parts) > 1 else str(idx)
                mapping[idx] = name
            except Exception:
                continue
    return mapping


def load_replace_map(filepath: str = 'db与df字段对应表.csv') -> Dict[str, str]:
    """读取字段名替换表，返回字典 old->new。"""
    mapping: Dict[str, str] = {}
    if not os.path.exists(filepath):
        print(f"Replace map file not found: {filepath}, continuing without it.")
        return mapping

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def find_all_jpgs(root_dir: str) -> List[str]:
    """递归查找 root_dir 下所有 jpg 文件，返回文件路径列表（跳过已存在的 .csv 结果）。"""
    results: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.jpg'):
                full = os.path.join(dirpath, fn)
                if not os.path.exists(full + '.csv'):
                    results.append(full)
    return results


def render_as_image(a: mx.nd.NDArray) -> np.ndarray:
    """将 MXNet NDArray 转为 uint8 numpy 图像。"""
    try:
        img = a.asnumpy()
        return img.astype(np.uint8)
    except Exception:
        return np.array([])


def segment_one_image_ade(img_path: str,
                          model,
                          ctx: mx.Context,
                          col_map: Dict[int, str],
                          save_visual: bool = True) -> Optional[pd.Series]:
    """对单张图片做分割，返回按 col_map 重命名的 Series（每类占比），并保存 CSV 与可视化。

    如果已存在对应 CSV 文件则跳过并返回 None。
    """
    try:
        if os.path.exists(img_path.replace('.jpg', '.csv')):
            print(f"Skipping existing result for {img_path}")
            return None

        img = image.imread(img_path)
        base = render_as_image(img)

        data = test_transform(img, ctx=ctx)
        output = model.predict(data)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        # 统计每个类别的像素比例（ADE20K 类别通常为 0..149）
        h, w = predict.shape
        total = h * w
        pred = []
        for i in range(0, 150):
            pred.append((predict == i).sum() / total)

        series = pd.Series(pred)
        series.index = [col_map.get(i, str(i)) for i in range(150)]

        # 可视化并保存
        if save_visual:
            try:
                vis_path = img_path.replace('.jpg', '_v.jpg')
                if os.path.exists(vis_path):
                    os.remove(vis_path)
                mask = get_color_pallete(predict, 'ade20k')
                plt.figure(figsize=(10, 5))
                plt.imshow(mask)
                plt.axis('off')
                plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Visualization failed for {img_path}: {e}")

        # 将结果保存为 CSV（加上 pid, heading 占位列以兼容原脚本格式）
        csv_path = img_path.replace('.jpg', '.csv')
        cols = ['pid', 'heading'] + list(series.index)
        df = pd.DataFrame(columns=cols)
        row = {'pid': 'pid', 'heading': 'heading'}
        row.update(series.to_dict())
        df = df.append(row, ignore_index=True)
        df.to_csv(csv_path, index=False)

        print(f"Completed segmentation for {img_path}")
        return series
    except Exception as e:
        print(f"Segmentation failed for {img_path}: {e}")
        return None


def main(root_dir: str = r'D:\a',
         ade_class_file: str = 'ADE20K_Class.csv',
         gpu_id: int = 0):
    """主流程：加载模型与映射，批量对目录下图片进行分割并保存结果。"""
    ctx = prepare_context(gpu_id)
    model = load_model(ctx)
    col_map = load_ade_class_map(ade_class_file)

    img_paths = find_all_jpgs(root_dir)
    print(f"Found {len(img_paths)} images to process")

    for img_path in img_paths:
        segment_one_image_ade(img_path, model, ctx, col_map, save_visual=True)


if __name__ == '__main__':
    # 调整 root_dir 与 ade_class_file 到你的实际路径后运行
    ROOT_DIR = r'D:\your_image_folder'
    ADE_CLASS_FILE = 'ADE20K_Class.csv'
    main(ROOT_DIR, ADE_CLASS_FILE, gpu_id=0)