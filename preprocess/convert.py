import pandas as pd
import numpy as np
from PIL import Image

csv_path = "preprocess/data/water_quality_data_interpolated.csv"
features = ["pH值", "溶解氧", "电导率", "浑浊度", "氨氮", "耗氧量"]

df = pd.read_csv(
    (csv_path),
    parse_dates=[0],
)  # 格式化采样日期列为 Timestamp
df.set_index("采样日期", inplace=True)  # 将采样日期列设置为索引列
# 将 DataFrame 索引提取去重，得到 datetime64[s] 数组
date_array = np.unique(df.index.to_numpy().astype("datetime64[s]"))

# 对日期数组进行升序排序（从早到晚）
date_array_sorted = np.sort(date_array)

tensor = np.zeros((20, len(date_array_sorted) - 20 + 1, 64, 64), dtype=np.uint8)

for index, feature in enumerate(features):
    for group_idx in range(len(date_array_sorted) - 20 + 1):
        for frame_idx in range(20):
            filename = f"preprocess/kriging/{index}_{pd.to_datetime(str(date_array_sorted[group_idx + frame_idx])).strftime("%Y%m%d_%H%M%S")}.png"

            img = Image.open(filename)
            img_array = np.array(img)

            tensor[frame_idx, group_idx] = img_array

    print("shape:", tensor.shape)
    np.savez_compressed(f"water_quality_{index}.npz", tensor)

print("finished.")
