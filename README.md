# BaiduPanoramaSpider 🌐🕵️‍♂️

一个用于学习和研究用途的百度街景（全景）图片爬取、坐标转换、畸变矫正与语义分割的工具集合。

> 本仓库包含：爬取百度街景瓦片、WGS84/GCJ02/BD09/BD09MC 坐标转换、等距全景到透视投影矫正、以及用于语义分割的脚本样例。

## 重要免责声明 ⚠️

- 本项目仅供学习与研究用途。使用本代码对第三方服务进行抓取前，用户必须自行确认并遵守目标服务的使用条款、当地法律及隐私法规。
- 作者未授权任何人利用本项目进行未经许可的大规模抓取、商业化利用或侵害隐私的活动。对因使用本项目产生的任何法律或经济后果，作者不承担责任。
- 若发现本项目被用于明显违法/侵权用途，可通过 repo issue 或联系邮箱提出通知。

## 功能概览 ✨

- 下载百度街景全景图（Baidu Panorama）并合并切片为整图。
- 坐标转换工具：WGS84 <-> GCJ02 <-> BD09 <-> BD09MC（位于 `coordinatetransformer.py`）。
- 全景图畸变矫正：等距圆柱（2:1 全景）转透视投影（`panorama_rectify.py`）。
- 语义分割脚本示例（使用 MXNet / GluonCV）：见 `segrepano-*` 文件（仅示例，依赖较重）。
- 帮助脚本入口：`main.py`（调用 `imgspider.py` 中的下载器）。

## 目录结构（摘要） 🗂️

- `imgspider.py` - 主爬虫与工具函数（下载、合并、CSV 批量流程）。
- `coordinatetransformer.py` - 坐标转换函数集。
- `panorama_rectify.py` - 全景畸变矫正工具与示例 `demo_rectify()`。
- `segrepano-*.py` - 语义分割示例脚本（CPU/GPU 版本）。
- `resources/` - 示例 CSV 与若干已下载/矫正的图片样本。
- `akandsk.txt` - 建议用于存放 AK/SK（本文件在 `.gitignore` 中默认被忽略，便于本地存放密钥）。

## 先决条件与安装 🔧

建议在虚拟环境中运行（venv / conda）。以下是最小且常用的 Python 包：

- Python 3.8+
- requests
- Pillow
- numpy
- pandas
- matplotlib (可选，用于可视化)

语义分割相关（可选，依赖较大，mxnet对于新的gpu而言，在Windows环境下非常难以安装，Linux环境中比较好配置）：

- mxnet (cpu 或 gpu 版本)
- gluoncv

## 配置 AK/SK 🔑

1. `main.py` 中写入ak与sk,传入 `imgspider`。

注意：`.gitignore` 中默认忽略 `*.txt`，因此 `akandsk.txt` 不会被提交（便于本地保存密钥）。

## 快速开始 — 下载街景（示例） ⬇️

仓库中提供 `main.py`，演示两种调用：

1. 使用 AK/SK 有助于更好的坐标转换：

```powershell
# 编辑 main.py，或在 akandsk.txt 放入你的密钥，然后运行：
python .\main.py
```

1. 不使用 AK/SK 坐标转换会有一定误差，笔者实测约为0.5-1米：

```powershell
# main.py 中有不使用 ak 的示例调用
python .\main.py
```

CSV 格式：示例位于 `resources/example.csv`，格式为两列 X,Y（WGS84 经度/纬度）。

## 全景畸变矫正（示例） 🔁

项目内 `panorama_rectify.py` 包含 `PanoramaRectifier` 与 `demo_rectify()`，用来把等距圆柱全景图生成透视视图。

示例运行（PowerShell）：

```powershell
python .\panorama_rectify.py
```

运行后脚本会处理 `resources/downloadPic` 并输出到 `resources/rectified`（或按脚本内路径配置）。

## 语义分割（示例，非必须） 🧠

脚本 `segrepano-*.py` 是分割流程示例，使用 GluonCV 的预训练模型。注意：这些脚本要求安装 MXNet 与 GluonCV，且可能需要 GPU。请参考各自脚本头部的注释以配置环境。

运行建议：

```powershell
# 安装（示例，按官方说明选择合适的 mxnet 版本）
pip install mxnet
pip install gluoncv
python .\segrepano-cpu-deeplabv3-resnet101-citys.py
```

这些脚本包含大量示例/检查逻辑，请在了解各依赖及显存限制后运行。

## 常见问题与排错 🐞

- 如果下载失败，检查网络和目标接口是否被限制。
- 若坐标转换结果异常，确认输入 CSV 是 WGS84（经度在前，纬度在后）。
- 语义分割报错通常为依赖未安装或 GPU/驱动不匹配。

## 安全与合规建议 🔒

- 请勿在未经许可的情况下进行大规模爬取或商业使用。尊重数据提供方的使用条款和隐私政策。
- 密钥请仅保存在本地（`akandsk.txt`），并确认 `.gitignore` 能正确忽略它们。

## 贡献与联系 🤝

欢迎提 issue 或 PR 来改进功能、修复 bug 或补充使用文档。此仓库用于研究与教学交流。

---

作者: yihanchangbanqingchun
日期: 2025-08-22

## 致谢与参考文献 🙏📚

本项目融合并改进了若干开源项目中的方法，特此致谢：

- CoordinatesConverter（作者: dickwxyz）
  - 仓库: [dickwxyz/CoordinatesConverter](https://github.com/dickwxyz/CoordinatesConverter)
  - 说明: 提供了丰富的坐标转换方法（WGS84/GCJ02/BD09 等），在本项目中参考并复用其算法实现。

- BaiduPanoramaSpider（作者: Zhen3r）
  - 仓库: [Zhen3r/BaiduPanoramaSpider](https://github.com/Zhen3r/BaiduPanoramaSpider)
  - 说明: 原始项目包含百度街景下载、合并与流程控制，本仓库在其基础上做了若干改进并合并了坐标转换/畸变矫正/语义分割示例。

本仓库的改动要点（简要说明）：

- 将 `CoordinatesConverter` 的坐标转换方法参考并整合进本项目的转换流程，以提升坐标转换的准确性与可维护性。
- 在 `BaiduPanoramaSpider` 的基础上，限定并改进了使用 AK 的坐标转换流程。
- 新增/集成了全景畸变矫正（`panorama_rectify.py`）与语义分割示例脚本（`segrepano-*`），便于后续分析与可视化处理。

再次感谢两位开源作者的贡献与许可，使得本项目能够在其基础上进行学习与实验性开发。
