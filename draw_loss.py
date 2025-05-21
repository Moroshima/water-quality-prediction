import matplotlib.pyplot as plt
import numpy as np

# 原始数据
epochs = np.arange(1, 21)
loss = [
    0.1027,
    0.0312,
    0.0301,
    0.0298,
    0.0297,
    0.0296,
    0.0295,
    0.0295,
    0.0294,
    0.0294,
    0.0294,
    0.0294,
    0.0293,
    0.0292,
    0.0293,
    0.0292,
    0.0293,
    0.0293,
    0.0292,
    0.0292,
]
val_loss = [
    0.1041,
    0.0926,
    0.0466,
    0.0309,
    0.0297,
    0.0296,
    0.0296,
    0.0294,
    0.0296,
    0.0295,
    0.0294,
    0.0293,
    0.0293,
    0.0294,
    0.0293,
    0.0293,
    0.0293,
    0.0293,
    0.0292,
    0.0292,
]
learning_rate = [0.001] * 18 + [0.0001] * 2  # 已简化科学计数法表示

# 创建画布
plt.figure(figsize=(12, 6))

# 绘制损失曲线（左轴）
ax1 = plt.gca()
ln1 = ax1.plot(epochs, loss, "b-", label="Training Loss", marker="o", markersize=4)
ln2 = ax1.plot(
    epochs, val_loss, "r--", label="Validation Loss", marker="s", markersize=4
)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.7)

# 创建第二个y轴显示学习率
ax2 = ax1.twinx()
ln3 = ax2.plot(epochs, learning_rate, "g-.", label="Learning Rate", linewidth=2)
ax2.set_ylabel("Learning Rate", fontsize=12)
ax2.set_yscale("log")  # 使用对数坐标更好显示数量级变化

# 合并图例
lines = ln1 + ln2 + ln3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

# 设置其他参数
plt.title("Training Dynamics (Loss & Learning Rate)", fontsize=14, pad=20)
plt.xlim(0, 21)
ax1.set_ylim(0.028, 0.11)  # 聚焦主要变化区域
ax2.set_ylim(1e-5, 1e-2)  # 对数坐标范围

# 突出显示学习率变化点
plt.annotate(
    "LR reduced to 0.0001",
    xy=(19, 0.0001),
    xytext=(15, 0.0002),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
    fontsize=10,
)

plt.tight_layout()
plt.show()
