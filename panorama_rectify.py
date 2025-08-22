import numpy as np
import os
from PIL import Image
import math

class PanoramaRectifier:
    """全景图畸变矫正器"""
    
    def __init__(self, fov=90, output_width=800, output_height=600):
        """
        :param fov: 视场角（度），默认90度
        :param output_width: 输出图像宽度
        :param output_height: 输出图像高度
        """
        self.fov = fov
        self.output_width = output_width
        self.output_height = output_height
        
    def bilinear_interpolate(self, img_array, x, y):
        """双线性插值"""
        h, w = img_array.shape[:2]
        
        # 边界检查
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        
        # 插值权重
        wx = x - x0
        wy = y - y0
        
        if len(img_array.shape) == 3:
            # 彩色图像
            result = np.zeros((y.shape[0], y.shape[1], img_array.shape[2]), dtype=img_array.dtype)
            for c in range(img_array.shape[2]):
                result[:, :, c] = (
                    img_array[y0, x0, c] * (1 - wx) * (1 - wy) +
                    img_array[y0, x1, c] * wx * (1 - wy) +
                    img_array[y1, x0, c] * (1 - wx) * wy +
                    img_array[y1, x1, c] * wx * wy
                )
        else:
            # 灰度图像
            result = (
                img_array[y0, x0] * (1 - wx) * (1 - wy) +
                img_array[y0, x1] * wx * (1 - wy) +
                img_array[y1, x0] * (1 - wx) * wy +
                img_array[y1, x1] * wx * wy
            )
        
        return result.astype(img_array.dtype)
        
    def equirectangular_to_perspective(self, panorama_img, yaw=0, pitch=0):
        """
        将等距圆柱投影（全景图）转换为透视投影
        :param panorama_img: 输入的全景图像(numpy array)
        :param yaw: 水平旋转角度（度）
        :param pitch: 垂直旋转角度（度）
        :return: 透视投影图像
        """
        pano_height, pano_width = panorama_img.shape[:2]
        
        # 创建输出图像坐标网格
        x_out, y_out = np.meshgrid(
            np.arange(self.output_width),
            np.arange(self.output_height)
        )
        
        # 将输出坐标转换为相机坐标系 (-1 到 1)
        x_cam = (x_out - self.output_width / 2) / (self.output_width / 2)
        y_cam = -(y_out - self.output_height / 2) / (self.output_height / 2)
        
        # 计算焦距
        f = 1 / math.tan(math.radians(self.fov / 2))
        
        # 3D射线方向向量
        x_3d = x_cam
        y_3d = y_cam * (self.output_height / self.output_width)
        z_3d = np.full_like(x_3d, f)
        
        # 归一化
        norm = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
        x_3d /= norm
        y_3d /= norm
        z_3d /= norm
        
        # 应用旋转
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        # 绕Y轴旋转(yaw)
        x_rot = x_3d * math.cos(yaw_rad) + z_3d * math.sin(yaw_rad)
        y_rot = y_3d
        z_rot = -x_3d * math.sin(yaw_rad) + z_3d * math.cos(yaw_rad)
        
        # 绕X轴旋转(pitch)
        x_final = x_rot
        y_final = y_rot * math.cos(pitch_rad) - z_rot * math.sin(pitch_rad)
        z_final = y_rot * math.sin(pitch_rad) + z_rot * math.cos(pitch_rad)
        
        # 转换为全景图坐标
        longitude = np.arctan2(x_final, z_final)
        latitude = np.arcsin(np.clip(y_final, -1, 1))
        
        # 映射到图像像素坐标
        u = (longitude / (2 * math.pi) + 0.5) * pano_width
        v = (-latitude / math.pi + 0.5) * pano_height
        
        # 使用双线性插值
        result = self.bilinear_interpolate(panorama_img, u, v)
        
        return result
    
    def process_single_image(self, input_path, output_path, views=None):
        """
        处理单张全景图，生成多个视角
        :param input_path: 输入图像路径
        :param output_path: 输出路径（不含扩展名）
        :param views: 视角列表，格式为[(yaw, pitch, suffix), ...]
        """
        if views is None:
            # 默认生成4个主要方向视角
            views = [
                (90, 0, "front"),      # 正前方
                (180, 0, "right"),     # 右侧
                (-90, 0, "back"),     # 后方
                (0, 0, "left"),     # 左侧
                (0, -30, "down"),     # 向下30度
                (0, 30, "up")         # 向上30度
            ]
        
        try:
            # 读取全景图
            img_pil = Image.open(input_path)
            img = np.array(img_pil)
            
            print(f"处理图像: {input_path}")
            print(f"原始尺寸: {img.shape[1]}x{img.shape[0]}")
            
            # 生成各个视角
            for yaw, pitch, suffix in views:
                rectified = self.equirectangular_to_perspective(img, yaw, pitch)
                output_file = f"{output_path}_{suffix}.jpg"
                
                # 转换回PIL图像并保存
                result_pil = Image.fromarray(rectified)
                result_pil.save(output_file, quality=90)
                print(f"已生成视角 {suffix}: {output_file}")
                
        except Exception as e:
            print(f"处理图像时出错 {input_path}: {e}")
    
    def process_folder(self, input_folder, output_folder):
        """
        批量处理文件夹中的全景图
        :param input_folder: 输入文件夹路径
        :param output_folder: 输出文件夹路径
        """
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # 遍历输入文件夹
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"在文件夹 {input_folder} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_folder, filename)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, name_without_ext)
            
            print(f"[{i}/{len(image_files)}] ", end="")
            self.process_single_image(input_path, output_path)
            print()

def demo_rectify():
    """演示函数：矫正下载的街景图片"""
    # 配置参数
    input_folder = "resources/downloadPic"  # 你的街景图片文件夹
    output_folder = "resources/rectified"   # 矫正后图片输出文件夹
    
    # 创建矫正器
    rectifier = PanoramaRectifier(
        fov=90,          # 90度视场角，接近人眼视角
        output_width=800,  # 输出宽度
        output_height=600  # 输出高度
    )
    
    print("开始全景图畸变矫正...")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print("-" * 50)
    
    # 批量处理
    rectifier.process_folder(input_folder, output_folder)
    
    print("畸变矫正完成！")
    print(f"矫正后的图片保存在: {output_folder}")

if __name__ == "__main__":
    # 运行演示
    demo_rectify()
