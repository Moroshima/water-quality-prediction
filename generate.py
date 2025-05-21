from PIL import Image, ImageDraw
import math
import random

# 参数设置
width, height = 64, 64  # 画布尺寸
square_size = 8  # 正方形边长
speed = 4  # 移动速度

for group in range(1000):

    # 随机初始位置
    x = random.randint(0, width - square_size)
    y = random.randint(0, height - square_size)
    # x = (width - square_size) / 2
    # y = (height - square_size) / 2

    # 随机初始方向（0到360度）
    theta = random.uniform(0, 2 * math.pi)
    vx = speed * math.cos(theta)
    vy = -speed * math.sin(theta)  # y轴向下，速度取反

    # 生成20张图像
    for frame in range(20):
        # 更新位置
        x += vx
        y += vy

        # 边界碰撞检测及反弹
        if x < 0:  # 左边缘
            vx = abs(vx)
            x = 0
        elif x + square_size > width:  # 右边缘
            vx = -abs(vx)
            x = width - square_size

        if y < 0:  # 上边缘
            vy = abs(vy)
            y = 0
        elif y + square_size > height:  # 下边缘
            vy = -abs(vy)
            y = height - square_size

        # 绘制图像
        img = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [x, y, x + square_size, y + square_size], fill="white", outline="yellow"
        )
        img.save(f"generate/group_{group:04d}_frame_{frame:02d}.png")

print("finished.")
