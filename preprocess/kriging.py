import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from PIL import Image
import os

kriging_resolution = 64  # Kriging 插值分辨率

csv_path = "preprocess/data/water_quality_data_interpolated.csv"

df = pd.read_csv(
    (csv_path),
    parse_dates=[0],
)  # 格式化采样日期列为 Timestamp
df.set_index("采样日期", inplace=True)  # 将采样日期列设置为索引列
# 将 DataFrame 索引提取去重，得到 datetime64[s] 数组
date_array = np.unique(df.index.to_numpy().astype("datetime64[s]"))
# 使用字典存放每组 DataFrame
dfs = {
    np.datetime64(timestamp).astype("datetime64[s]"): data
    for timestamp, data in df.groupby(df.index)
}

features = ["pH值", "溶解氧", "电导率", "浑浊度", "氨氮", "耗氧量"]

# 获取每个水质参数的全局最小值和最大值
feature_min = {}
feature_max = {}
for feature in features:
    feature_min[feature] = df[feature].min()
    feature_max[feature] = df[feature].max()
np.save("feature_min_max.npy", (feature_min, feature_max))

# for specific_date in date_array:
#     # data_slice 是特定日期的 DataFrame 切片，其有效信息为：经度、纬度与 6 项水质指标，为三维数据，以经纬度为平面坐标，水质指标为高度坐标
#     data_slice = dfs[specific_date]

#     # 5点六参数坐标
#     X = data_slice["采样经度"].values
#     Y = data_slice["采样纬度"].values

#     for index, feature in enumerate(features):
#         Z = data_slice[feature].values

#         # 定义插值网格
#         gridx = np.linspace(np.min(X), np.max(X), kriging_resolution)
#         gridy = np.linspace(np.min(Y), np.max(Y), kriging_resolution)

#         # 检查 Z 数组中的所有值是否全都相等
#         if np.all(Z == Z[0]):
#             # 如果所有Z值相等，直接构造一个全是 Z[0] 的网格作为估计值，方差设为 0
#             z = np.full((len(gridx), len(gridy)), Z[0])
#             ss = np.zeros((len(gridx), len(gridy)))
#         else:
#             # 创建 Kriging 模型并进行拟合后执行“线性”插值
#             z, ss = OrdinaryKriging(X, Y, Z, variogram_model="linear").execute(
#                 "grid", gridx, gridy
#             )  # 将点与特征输入插值函数，得到插值后的矩阵 z，即水质平面分布数据，即图片

#         # 归一化
#         min_val = feature_min[feature]
#         max_val = feature_max[feature]

#         # 处理所有值相等的情况（避免除以零）
#         if min_val == max_val:
#             normalized_z = np.full_like(z, min_val)
#         else:
#             # 归一化并缩放到0-255范围
#             normalized_z = 255 * (z - min_val) / (max_val - min_val)
#             # 确保值在0-255范围内
#             # normalized_z = np.clip(normalized_z, 0, 255)

#         image_data = normalized_z.astype(np.uint8)
#         # 创建并保存灰度图像
#         img = Image.fromarray(image_data)
#         img.save(
#             f"preprocess/kriging/{index}_{pd.to_datetime(str(specific_date)).strftime("%Y%m%d_%H%M%S")}.png"
#         )

# 在现有代码的基础上添加以下部分
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 获取第x天的日期
date = date_array[0]
data_slice = dfs[date]

# 提取index=0的特征(pH值)
feature = features[0]
X = data_slice["采样经度"].values
Y = data_slice["采样纬度"].values
Z = data_slice[feature].values

# 重新计算插值网格
gridx = np.linspace(np.min(X), np.max(X), kriging_resolution)
gridy = np.linspace(np.min(Y), np.max(Y), kriging_resolution)

# 执行Kriging插值
if np.all(Z == Z[0]):
    z = np.full((len(gridx), len(gridy)), Z[0])
else:
    z, _ = OrdinaryKriging(X, Y, Z, variogram_model="linear").execute(
        "grid", gridx, gridy
    )

# 创建图形
plt.figure(figsize=(12, 10))
plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
ax = plt.gca()

# 绘制插值结果
contour = ax.contourf(gridx, gridy, z.T, 20, cmap="viridis")
plt.colorbar(contour, label=feature, shrink=0.8)

# 标注采样点
scatter = ax.scatter(X, Y, c="red", label="采样点", zorder=5)
ax.ticklabel_format(useOffset=False, style="plain")

# 在采样点旁添加数值标注
for i, (x, y, z_val) in enumerate(zip(X, Y, Z)):
    plt.annotate(
        f"{z_val:.2f}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10
    )

# 设置坐标轴
ax.set_xlabel("经度", fontsize=14)
ax.set_ylabel("纬度", fontsize=14)
ax.set_title(
    f'{pd.to_datetime(str(date)).strftime("%Y年%m月%d日")} - {feature}分布',
    fontsize=16,
)

# 设置坐标轴刻度
ax.xaxis.set_major_locator(MultipleLocator(0.001))
ax.yaxis.set_major_locator(MultipleLocator(0.001))

# 添加图例
ax.legend(loc="upper right", fontsize=12)

# 添加比例尺和指北针
ax.text(0.95, 0.02, "N", transform=ax.transAxes, fontsize=20, ha="center")
ax.arrow(
    0.95,
    0.05,
    0,
    0.03,
    transform=ax.transAxes,
    head_width=0.02,
    head_length=0.02,
    fc="k",
    ec="k",
)

plt.tight_layout()  # 调整布局
plt.show()
