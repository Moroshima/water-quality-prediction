import numpy as np
from PIL import Image

# 初始化目标张量 (20, 1000, 64, 64)
tensor = np.zeros((20, 1000, 64, 64), dtype=np.uint8)

# 遍历每个时间帧
for frame_idx in range(20):
    # 遍历每个组
    for group_idx in range(1000):
        # 生成文件名（根据实际路径可能需要添加目录前缀）
        filename = f"generate/group_{group_idx:04d}_frame_{frame_idx:02d}.png"

        # 读取图像文件
        img = Image.open(filename)

        # 转换为NumPy数组
        img_array = np.array(img)
        # print(img_array.shape)

        # 分割为8x8的块并计算平均值
        blocks = img_array.reshape(8, 8, 8, 8)
        block_means = blocks.mean(axis=(1, 3)).round().astype(int)

        # 输出结果矩阵
        # print(block_means.shape)

        block_kron = np.kron(block_means, np.ones((8, 8), dtype=int))
        # print(block_kron.shape)

        # 存储到张量中
        tensor[frame_idx, group_idx] = block_kron

print("shape:", tensor.shape)
np.savez_compressed("water_quality.npz", tensor)
