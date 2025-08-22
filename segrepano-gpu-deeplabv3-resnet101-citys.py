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


def require_cuda_or_raise() -> None:
    """确保 CUDA 可用，否则抛出异常。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")


def prepare_context() -> mx.Context:
    """返回用于计算的 MXNet 上下文（优先 GPU）。"""
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    print(f"Using context: {ctx}")
    return ctx


def prepare_transform() -> transforms.Compose:
    """构造并返回图像预处理变换（ToTensor + Normalize）。"""
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    return transform_fn


def load_model(ctx: mx.Context):
    """加载预训练的 deeplab_resnet101_citys 模型并返回它。

    参数:
        ctx: 计算上下文
    返回:
        已加载的模型实例
    """
    model = gluoncv.model_zoo.get_model('deeplab_resnet101_citys', pretrained=True, ctx=ctx)
    return model


def check_and_update_csv(csv_file_path: str, output_folder_path: str) -> pd.DataFrame:
    """检查 CSV 中标记为已处理的 PID 是否真的存在输出文件；若不存在则回退状态。

    返回更新后的 DataFrame（并且会写回 CSV）。
    """
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    if 'processed' not in df.columns:
        df['processed'] = 0

    for index, row in df.iterrows():
        pid = str(row['pid'])
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


def process_image(file_path: str,
                  file_pid: str,
                  model,
                  transform_fn,
                  ctx: mx.Context,
                  output_folder: str) -> bool:
    """处理单张图片：读取、预处理、预测、生成天空二值图并保存。

    返回是否成功处理（True 表示成功）。
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
    print(f"Image preprocessed and moved to context ({time.time() - start_time:.2f}s)")

    # 预测
    try:
        start_time = time.time()
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        print(f"Prediction done ({time.time() - start_time:.2f}s)")
    except Exception as exc:
        print(f"Model prediction failed for {file_path}: {exc}")
        return False

    # 识别天空（Cityscapes 标签中 10 为 sky）
    sky = (predict == 10)
    sky_image = np.zeros_like(predict, dtype=np.uint8)
    sky_image[sky] = 255

    # 保存结果
    try:
        output_path = os.path.join(output_folder, os.path.basename(file_path).replace('_panorama.jpg', '_sky.png'))
        plt.imsave(output_path, sky_image, cmap='gray')
        print(f"Saved sky image: {output_path}")
    except Exception as exc:
        print(f"Failed to save result for {file_path}: {exc}")
        return False

    return True


def main(input_folder: str, output_folder: str, csv_path: str) -> None:
    """主流程：校验环境、加载模型、检查 CSV 并批量处理图片。"""
    require_cuda_or_raise()

    # 基础准备
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    ctx = prepare_context()
    transform_fn = prepare_transform()
    model = load_model(ctx)

    # 读取并校验 CSV
    df = check_and_update_csv(csv_path, output_folder)
    pids = df['pid'].astype(str).tolist()
    processed_pids = df[df['processed'] == 1]['pid'].astype(str).tolist()

    total_files = len(pids)
    processed_files = len(processed_pids)

    # 输出目录中已有的 PNG 提示
    output_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    output_pids = [f.split('_')[0] for f in output_files]
    for pid in output_pids:
        if pid not in processed_pids:
            print(f"Warning: Output file with PID {pid} found but not marked as processed in CSV.")

    # 遍历输入文件夹并处理
    with tqdm(total=total_files, initial=processed_files, desc="处理进度") as pbar:
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith('.jpg'):
                continue

            file_pid = filename.split('_')[0]
            if file_pid not in pids or file_pid in processed_pids:
                continue

            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            ok = process_image(file_path, file_pid, model, transform_fn, ctx, output_folder)
            if ok:
                # 更新 DataFrame 并写回 CSV
                df.loc[df['pid'] == file_pid, 'processed'] = 1
                processed_pids.append(file_pid)
                df.to_csv(csv_path, index=False)
                pbar.update(1)

    print("Batch processing completed.")


if __name__ == '__main__':
    # 请根据实际环境修改下面的路径
    INPUT_FOLDER = r'D:\wuhan_rd_pano\processed_files'
    OUTPUT_FOLDER = r'D:\wuhan_rd_pano\sky_files'
    CSV_PATH = r'D:\街景全景_武汉\sunglare\pano_nr10inrd50_2_jishu.csv'

    main(INPUT_FOLDER, OUTPUT_FOLDER, CSV_PATH)