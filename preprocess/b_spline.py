import numpy as np
import matplotlib.pyplot as plt

# from scipy.interpolate import lagrange  # LAGRANGE
from scipy.interpolate import BSpline
import pandas as pd

interpolation_freq = "6h"

input_df = pd.read_csv("preprocess/data/water_quality_data_cleaned.csv")
output_df = pd.DataFrame()

# 对每个点位分别进行插值
for i in range(5):
    point_name = f"白洋湾金墅水源地{i+1}"
    # 点位对应的 DataFrame
    df = input_df[input_df["点位名称"] == point_name]

    date_start = pd.to_datetime(df["采样日期"].values[0])
    date_end = pd.to_datetime(df["采样日期"].values[-1])

    date_range = pd.date_range(start=date_start, end=date_end, freq=interpolation_freq)

    # 准备用于存放插值后 b_spline_df 结构
    b_spline_df = pd.DataFrame()
    b_spline_df["采样日期"] = date_range
    b_spline_df.set_index("采样日期", inplace=True)
    b_spline_df["点位名称"] = point_name
    b_spline_df["采样经度"] = df["采样经度"].values[0]
    b_spline_df["采样纬度"] = df["采样纬度"].values[0]

    # 创建一个与原始数据时间对应的索引数组
    x = np.arange(len(df))

    # 生成插值后数据对应的细分x值数组
    x_new = np.linspace(x[0], x[-1], len(date_range))

    # 设置B样条的次数
    k = 3
    # 创建B样条的节点向量
    t = np.concatenate(
        ([x[0]] * k, np.linspace(x[0], x[-1], len(x) - k + 1), [x[-1]] * k)
    )

    parameters = ["pH值", "溶解氧", "电导率", "浑浊度", "氨氮", "耗氧量"]
    parameters_precision = [1, 2, 0, 0, 2, 2]  # 数据精度（小数点后n位）

    # 逐数据插值
    for index, value in enumerate(parameters):
        y = df[value].values
        # poly = lagrange(x, y)  # LAGRANGE 拉格朗日多项式
        bspl = BSpline(t, y, k)

        # 插值
        # y_new = poly(x_new)  # LAGRANGE
        y_new = bspl(x_new)

        b_spline_df[value] = np.around(y_new, decimals=parameters_precision[index])

        plt.figure(figsize=(12, 6))
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]

        # 将x轴转换为日期
        original_dates = pd.to_datetime(df["采样日期"])
        interpolated_dates = b_spline_df.index

        # 绘制原始数据点
        plt.scatter(original_dates, y, color="red", label="原始数据点", zorder=5)

        # 绘制插值曲线
        # plt.plot(interpolated_dates, y_new, "b-", label="拉格朗日插值曲线")  # LAGRANGE
        plt.plot(interpolated_dates, y_new, "b-", label="B样条插值曲线")

        # plt.title(f"{point_name} {value} - 拉格朗日插值与原始数据对比")  # LAGRANGE
        plt.title(f"{point_name} {value} - 三次B样条插值与原始数据对比")
        plt.xlabel("采样日期")
        plt.ylabel(value)
        plt.legend()
        plt.grid(True)

        # 调整x轴标签显示
        plt.gcf().autofmt_xdate()

        # 保存图像
        plt.savefig(
            # f"preprocess/graphs/{point_name}_{value}_拉格朗日插值对比.png",  # LAGRANGE
            f"preprocess/graphs/{point_name}_{value}_三次B样条插值对比.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 显示DataFrame的头部
    # print(b_spline_df.head())

    output_df = pd.concat([output_df, b_spline_df])

# print(output_df.sort_values(by=["采样日期", "点位名称"]))
output_df.to_csv(
    "preprocess/data/water_quality_data_interpolated.csv",
    date_format="%Y-%m-%dT%H:%M:%S",
)

print("finished.")
