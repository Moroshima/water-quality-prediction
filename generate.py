import numpy as np

canvas = np.zeros((8, 8))
points = []

# 生成随机位置
for _ in range(np.random.randint(1, 5 - 1)):
    x = np.random.randint(0, 10)
    y = np.random.randint(0, 10)
    canvas[x, y] = 1
    points.append((x, y))

print(canvas)
print(points)

# 生成随机大小
for _ in range(np.random.randint(1, 5 - 1)):
    size = np.random.randint(1, 5)
    x = np.random.randint(0, 10 - size)
    y = np.random.randint(0, 10 - size)
    canvas[x : x + size, y : y + size] = np.random.randint(1, 255)

# 1. 生成一个10，10的矩阵，并随即在其上生成1-3个原始数据点位置
# 2. 基于每个原始数据点生成一个x*x大小的区域（x是0到3之间的随机数），然后在区域内再随机生成0到(x*x)-1个数据点，这个区域里的每个数据点都是1-255之间的随机值
# (不必)且每个数据点的数值遵循从原始数据点往外递减的原则
# 3. 连续移动19次，生成20个不同的矩阵，每次移动将所有原始数据点随机往上下左右移动1-2个点，原始数据点之间的移动方向不必相同，且在移动完成后包含原始数据点在内的每个数据点数值需要随机增减1-255之间的随机值
# (不必)但依然要保证数据遵循从原始数据点往外递减的原则
# 如果上一步是往右移，那下一步只能往右移或者往右上、右下移动，以此类推
# (不必)在移动和调整数据点时增加20%的概率随机添加或删除数据点，并在特定情况下移除原始数据点。
# (不必)20%概率随机增减一个数据点的逻辑，如果减少时该数据点是最后一个数据点，且为原始数据点，将其从原始数据点列表或类似的地方移除