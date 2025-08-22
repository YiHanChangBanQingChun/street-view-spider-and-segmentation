import os
import time
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import torch
from typing import Tuple, List

# 保留原脚本输出的 cuDNN 版本信息（如果 environment 中有 torch）
try:
    print(torch.backends.cudnn.version())
except Exception:
    pass


def prepare_context() -> mx.Context:
    """返回 CPU 上下文（该脚本为 CPU 专用）。"""
    ctx = mx.cpu()
    print(f"Using context: {ctx}")
    return ctx


def prepare_transform() -> transforms.Compose:
    """构造输入图像的预处理变换。"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])


def load_model(ctx: mx.Context):
    """加载 CPU 上运行的预训练模型。"""
    model = gluoncv.model_zoo.get_model('deeplab_resnet101_citys', pretrained=True, ctx=ctx)
    return model


def check_and_update_csv(csv_file_path: str, output_folder_path: str) -> pd.DataFrame:
    """检查 CSV 中 processed 标记与输出文件是否一致，必要时回退标记并写回 CSV。

    返回更新后的 DataFrame。
    """
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    if 'processed' not in df.columns:
        df['processed'] = 0

    for index, row in df.iterrows():
        pid = row['pid']
        status = row.get('processed', 0)
        output_file_1 = os.path.join(output_folder_path, f"{pid}.png")
        output_file_2 = os.path.join(output_folder_path, f"{pid}_sky.png")
        if int(status) == 1:
            if not (os.path.exists(output_file_1) or os.path.exists(output_file_2)):
                df.at[index, 'processed'] = 0
                print(f"Updated status for pid {pid} to 0")

    df.to_csv(csv_file_path, index=False)
    print("CSV file updated")
    return df


def process_image(file_path: str, model, transform_fn, ctx: mx.Context, output_folder: str) -> bool:
    """读取并处理单张图片，生成 sky 二值图并保存。

    返回 True 表示处理并保存成功，False 表示出现错误。
    """
    try:
        start_time = time.time()
        img = mx.image.imread(file_path)
        print(f"Image read: {file_path} ({time.time() - start_time:.2f}s)")
    except Exception as exc:
        print(f"Error reading the image file {file_path}: {exc}")
        return False

    start_time = time.time()
    img = transform_fn(img)
    img = img.expand_dims(0).as_in_context(ctx)
    print(f"Image preprocessed ({time.time() - start_time:.2f}s)")

    try:
        start_time = time.time()
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        print(f"Prediction done ({time.time() - start_time:.2f}s)")
    except Exception as exc:
        print(f"Model prediction failed for {file_path}: {exc}")
        return False

    # sky label (Cityscapes) == 10
    sky = (predict == 10)
    sky_image = np.zeros_like(predict, dtype=np.uint8)
    sky_image[sky] = 255

    try:
        output_path = os.path.join(output_folder, os.path.basename(file_path).replace('_panorama.jpg', '_sky.png'))
        plt.imsave(output_path, sky_image, cmap='gray')
        print(f"Saved sky image: {output_path}")
    except Exception as exc:
        print(f"Failed to save result for {file_path}: {exc}")
        return False

    return True


def main(input_folder: str, output_folder: str, csv_path: str) -> None:
    """主流程：准备环境、加载模型、校验 CSV、逐张处理并更新 CSV。"""
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    ctx = prepare_context()
    transform_fn = prepare_transform()
    model = load_model(ctx)

    df = check_and_update_csv(csv_path, output_folder)
    pids = df['pid'].astype(str).tolist()
    processed_pids = df[df['processed'] == 1]['pid'].astype(str).tolist()

    total_files = len(pids)
    processed_files = len(processed_pids)

    output_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    output_pids = [f.split('_')[0] for f in output_files]
    for pid in output_pids:
        if pid not in processed_pids:
            print(f"Warning: Output file with PID {pid} found but not marked as processed in CSV.")

    with tqdm(total=total_files, initial=processed_files, desc="处理进度") as pbar:
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith('.jpg'):
                continue

            file_pid = filename.split('_')[0]
            if file_pid not in pids or file_pid in processed_pids:
                continue

            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            ok = process_image(file_path, model, transform_fn, ctx, output_folder)
            if ok:
                df.loc[df['pid'] == file_pid, 'processed'] = 1
                processed_pids.append(file_pid)
                df.to_csv(csv_path, index=False)
                pbar.update(1)

    print("Batch processing completed.")


if __name__ == '__main__':
    # 根据需要修改为本机路径
    INPUT_FOLDER = r'D:\wuhan_rd_pano\processed_files'
    OUTPUT_FOLDER = r'D:\wuhan_rd_pano\sky_files'
    CSV_PATH = r'D:\街景全景_武汉\sunglare\pano_nr10inrd50_2_jishu.csv'

    main(INPUT_FOLDER, OUTPUT_FOLDER, CSV_PATH)