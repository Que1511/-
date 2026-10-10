# -*- coding: utf-8 -*-
"""
基于 K-Means 的图像矢量量化实验
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无界面后端
import matplotlib.pyplot as plt
from PIL import Image
import os

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"C:\Users\baigw\Doubao\chats\2026-10-10\new-chat"
DATA_DIR = os.path.join(OUT_DIR, "data")

# ============ 第1部分：读取图片并向量化 ============
# 下载的图顶部有标题文字，裁剪掉只保留照片区域
raw = Image.open(os.path.join(DATA_DIR, "flower_raw.png"))
raw_arr = np.array(raw)
print("原始下载图形状:", raw_arr.shape)

# 裁剪掉顶部标题区域（白色背景 + 文字），只保留照片
# 标题大约在前 60 像素
if raw_arr.shape[0] > 100:
    flower = raw_arr[55:raw_arr.shape[0]-5, 5:raw_arr.shape[1]-5, :3]
else:
    flower = raw_arr[:, :, :3]

im = Image.fromarray(flower)
# 缩放到 640x427 以匹配实验指导书描述
im = im.resize((640, 427), Image.LANCZOS)
im.save(os.path.join(DATA_DIR, "flower.jpg"), quality=95)

image = np.array(im).astype(float) / 255.0
image = image[:, :, :3]
H, W, C = image.shape
print("图片形状:", image.shape)

X = image.reshape(-1, 3)
print("样本矩阵:", X.shape)

# 保存原图展示
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(image)
ax.axis('off')
ax.set_title("原图（彩色）", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "result_original.png"), dpi=150, bbox_inches='tight')
plt.close()
print("原图已保存")

# ============ 第2部分：手写 K-Means ============
def initCenter(dataSet, k, random_state=42):
    """从数据中随机选取 k 个样本作为初始中心"""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(dataSet.shape[0], size=k, replace=False)
    return dataSet[idx].copy()

def myKmeans(dataSet, k, random_state=42):
    """
    K-Means 主函数
    返回:
        classCenter: (k, 3) 簇中心
        clusterPoints: (n, 2) 第0列簇编号，第1列距离平方
    """
    n = dataSet.shape[0]
    # 初始化中心
    centers = initCenter(dataSet, k, random_state)
    clusterPoints = np.zeros((n, 2))
    clusterPoints[:, 0] = -1  # 初始簇编号设为 -1

    while True:
        # 第二步：分配样本 - 向量化计算距离
        # distances: (n, k)
        distances = np.sum((dataSet[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        newLabels = np.argmin(distances, axis=1)
        minDist = distances[np.arange(n), newLabels]

        # 判断收敛：所有样本簇分配不再变化
        if np.array_equal(newLabels, clusterPoints[:, 0].astype(int)):
            clusterPoints[:, 0] = newLabels
            clusterPoints[:, 1] = minDist
            break

        clusterPoints[:, 0] = newLabels
        clusterPoints[:, 1] = minDist

        # 第三步：更新中心
        for j in range(k):
            mask = newLabels == j
            if np.any(mask):
                centers[j] = dataSet[mask].mean(axis=0)
            # 空簇保留原中心不变

    return centers, clusterPoints

# ============ 第3部分：肘击法确定最优 K ============
print("\n=== 肘击法计算 SSE ===")
sse_list = []
k_range = range(1, 17)
for k in k_range:
    best_sse = float('inf')
    for seed in [42, 7, 123]:
        centers, cp = myKmeans(X, k, random_state=seed)
        sse = cp[:, 1].sum()
        if sse < best_sse:
            best_sse = sse
    sse_list.append(best_sse)
    print(f"K={k:2d}, SSE={best_sse:.4f}")

sse_arr = np.array(sse_list)

# 计算下降量，找肘部
diffs = np.diff(sse_arr)
# 计算下降率变化（二阶差分）
second_diff = np.diff(diffs)
# 找肘部：SSE下降从急剧变缓的点
# 简单方法：找最大曲率点
# 用 "Kneedle" 思路：归一化后找最大距离点
x_norm = (np.array(list(k_range)) - 1) / 15.0
y_norm = (sse_arr - sse_arr.min()) / (sse_arr.max() - sse_arr.min())
# 从 (0,1) 到 (1,0) 的直线
line_x = np.array([0, 1])
line_y = np.array([1, 0])
# 计算每个点到直线的距离
distances_to_line = np.abs(y_norm - (1 - x_norm)) / np.sqrt(2)
elbow_idx = np.argmax(distances_to_line)
best_k = elbow_idx + 1  # K从1开始

print(f"\n最优 K = {best_k}")

# 绘制 SSE-K 曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), sse_list, 'bo-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='r', linestyle='--', label=f'最优 K={best_k}')
ax.set_xlabel('K（聚类数）', fontsize=13)
ax.set_ylabel('SSE（误差平方和）', fontsize=13)
ax.set_title('肘击法确定最优 K（SSE-K 曲线）', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "result_elbow.png"), dpi=150, bbox_inches='tight')
plt.close()
print("肘击曲线已保存")

# ============ 第4部分：矢量量化重建与展示 ============
print("\n=== 矢量量化重建 ===")
def reconstruct(image, centers, labels):
    """用簇中心颜色替换像素，重建图像"""
    H, W, C = image.shape
    reconstructed = centers[labels.astype(int)].reshape(H, W, C)
    return reconstructed

# K=2
print("运行 K=2 ...")
centers2, cp2 = myKmeans(X, 2, random_state=42)
img_k2 = reconstruct(image, centers2, cp2[:, 0])

# K=最优K
print(f"运行 K={best_k} ...")
centers_best, cp_best = myKmeans(X, best_k, random_state=42)
img_best = reconstruct(image, centers_best, cp_best[:, 0])

# K=30
print("运行 K=30 ...")
centers30, cp30 = myKmeans(X, 30, random_state=42)
img_k30 = reconstruct(image, centers30, cp30[:, 0])

# 并排展示
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ['原图', 'K=2（2种代表颜色）', f'K=最优K（K={best_k}）', 'K=30（30种代表颜色）']
images = [image, img_k2, img_best, img_k30]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "result_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("对比图已保存")

print("\n=== 全部完成 ===")
print(f"最优K = {best_k}")
