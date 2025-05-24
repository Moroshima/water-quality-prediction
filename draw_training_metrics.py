import matplotlib.pyplot as plt
import numpy as np

history = np.load("model_training_history.npy", allow_pickle=True).item()
print(history.keys())

# 原始数据
loss = history["loss"]
val_loss = history["val_loss"]
learning_rate = history["learning_rate"]
root_mean_squared_error = history["root_mean_squared_error"]
val_root_mean_squared_error = history["val_root_mean_squared_error"]
epochs = range(1, len(loss) + 1)

# 创建画布
plt.figure(figsize=(12, 6))
plt.xticks(epochs)

# 绘制损失曲线（左轴）
ax1 = plt.gca()
ln1 = ax1.plot(epochs, loss, "b-", label="Training Loss", marker="o", markersize=4)
ln2 = ax1.plot(
    epochs, val_loss, "r--", label="Validation Loss", marker="s", markersize=4
)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)

# 创建第二个y轴显示学习率
ax2 = ax1.twinx()
ln3 = ax2.plot(epochs, learning_rate, "g-.", label="Learning Rate", linewidth=2)
ax2.set_ylabel("Learning Rate", fontsize=12)
ax2.set_yscale("log")  # 使用对数坐标更好显示数量级变化

# 合并图例
lines = ln1 + ln2 + ln3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")


# 设置其他参数
plt.title("Loss & Learning Rate", fontsize=14, pad=20)
ax2.set_ylim(1e-5, 1e-2)  # 对数坐标范围

plt.tight_layout()
plt.show()

# 创建新画布专门展示RMSE
plt.figure(figsize=(12, 6))
plt.xticks(epochs)

# 绘制RMSE曲线
plt.plot(
    epochs,
    root_mean_squared_error,
    "m-",
    label="Training RMSE",
    marker="o",
    markersize=6,
)
plt.plot(
    epochs,
    val_root_mean_squared_error,
    "c--",
    label="Validation RMSE",
    marker="D",
    markersize=5,
)

# 设置图表元素
plt.title("Root Mean Squared Error", fontsize=14, pad=20)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="best")

plt.tight_layout()
plt.show()
