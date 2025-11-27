# lunwen
"""
五种测试函数的可视化代码
每个函数生成一张组合图（包含3D曲面图和等高线图）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============= 函数定义 =============

def test_function_1(x, y):
    """函数1: 简单光滑函数（基准）"""
    z = (0.75 * np.exp(-((9*x-2)**2 + (9*y-2)**2)/4) +
         0.75 * np.exp(-((9*x+1)**2)/49 - (9*y+1)/10) +
         0.5 * np.exp(-((9*x-7)**2 + (9*y-3)**2)/4) -
         0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2))
    return z

def test_function_2(x, y):
    """函数2: 振荡函数（中等复杂度）"""
    r = np.sqrt(x**2 + y**2)
    z = np.exp(-0.5 * r) * np.sin(5 * r) + 0.3 * np.log(1 + r**2)
    z += 0.5 * np.sin(3*x) * np.cos(3*y)
    return z

def test_function_3(x, y):
    """函数3: 分段不连续函数（高复杂度）"""
    z = np.zeros_like(x, dtype=np.float64)
    r = np.sqrt(x**2 + y**2)
    # 区域1：[-3,3]×[-3,3]
    mask1 = (np.abs(x) <= 3) & (np.abs(y) <= 3)
    z[mask1] = np.exp(-0.5 * r[mask1]) * np.sin(5 * r[mask1]) + 0.3 * np.log(1 + r[mask1]**2)
    z[(x == 0) & (y == 0)] = 0
    # 区域2：|x|>3
    mask2 = np.abs(x) > 3
    z[mask2] = 2.5 * np.sign(x[mask2]) * np.exp(-np.abs(x[mask2])) * np.cos(2 * np.pi * y[mask2])
    return z

def test_function_4(x, y):
    """函数4: 多尺度混合函数（极高复杂度）"""
    z = np.zeros_like(x, dtype=np.float64)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # 特征1: 中心高斯山峰
    z += 3.0 * np.exp(-0.3 * r**2)
    
    # 特征2: 径向高频振荡
    z += 0.8 * np.sin(8 * r) * np.exp(-0.2 * r)
    
    # 特征3: 角向变化
    z += 1.5 * np.sin(4 * theta) * np.exp(-0.15 * r)
    
    # 特征4: 双曲面鞍点
    z += 0.5 * (x**2 - y**2) / (1 + 0.5*r**2)
    
    # 特征5: 局部尖峰
    peak1_dist = np.sqrt((x - 2)**2 + (y - 2)**2)
    peak2_dist = np.sqrt((x + 2)**2 + (y + 2)**2)
    z += 2.0 / (1 + 5*peak1_dist**2)
    z += 2.0 / (1 + 5*peak2_dist**2)
    
    # 特征6: 分段线性区域
    mask_corner = (x > 0) & (y < 0) & (np.abs(x) < 3) & (np.abs(y) < 3)
    z[mask_corner] += 1.5 * np.abs(x[mask_corner]) + 0.5 * np.abs(y[mask_corner])
    
    # 特征7: 高频棋盘格振荡
    mask_oscillate = (x < 0) & (y > 0) & (np.abs(x) < 3) & (np.abs(y) < 3)
    z[mask_oscillate] += 0.8 * np.sin(10*x[mask_oscillate]) * np.sin(10*y[mask_oscillate])
    
    return z

def test_function_5(x, y):
    """函数5: 病态函数（极端挑战）"""
    z = 1.0 / (1 + 25 * (x**2 + y**2))
    z += 0.5 * np.sin(20 * x) * np.exp(-y**2)
    z += 0.3 * np.tanh(5*x) * np.tanh(5*y)
    return z

# ============= 可视化函数 =============

def plot_function_3d(func, x_range, y_range, title, filename, resolution=100):
    """绘制3D曲面图"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D曲面
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9, 
                          linewidth=0, antialiased=True)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"已保存: {filename}")
    plt.close()

def plot_function_contour(func, x_range, y_range, title, filename, resolution=200):
    """绘制等高线图"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制等高线填充图
    contour = ax.contourf(X, Y, Z, levels=30, cmap=cm.viridis)
    
    # 绘制等高线
    contour_lines = ax.contour(X, Y, Z, levels=30, colors='black', 
                              alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # 添加颜色条
    fig.colorbar(contour, ax=ax, label='函数值 Z')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"已保存: {filename}")
    plt.close()

def plot_function_combined(func, x_range, y_range, title, filename, resolution=200):
    """绘制组合图：3D曲面 + 等高线"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 左图：3D曲面
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9, 
                           linewidth=0, antialiased=True)
    ax1.set_title('3D曲面图', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_zlabel('Z', fontsize=11)
    fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=25, pad=0.1)
    
    # 右图：等高线
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap=cm.viridis)
    contour_lines = ax2.contour(X, Y, Z, levels=30, colors='black', 
                                alpha=0.3, linewidths=0.5)
    ax2.set_title('等高线图', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2, label='函数值 Z', shrink=0.9, pad=0.1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 已保存: {filename}")
    plt.close()

# ============= 主程序 =============

if __name__ == "__main__":
    print("=" * 80)
    print("五种测试函数可视化")
    print("=" * 80)
    
    # 定义域
    x_range = [-5, 5]
    y_range = [-5, 5]
    
    # 函数列表
    functions = [
        (test_function_1, "函数1：光滑函数（基准）", "函数1_光滑函数"),
        (test_function_2, "函数2：振荡函数（中等复杂度）", "函数2_振荡函数"),
        (test_function_3, "函数3：分段不连续函数（高复杂度）", "函数3_分段不连续函数"),
        (test_function_4, "函数4：多尺度混合函数（极高复杂度）", "函数4_多尺度混合函数"),
        (test_function_5, "函数5：病态函数（极端挑战）", "函数5_病态函数")
    ]
    
    print("\n开始生成图片...\n")
    
    # 每个函数只生成一张组合图
    for func, title, name in functions:
        print(f"处理 {title}...")
        
        # 生成组合图（3D曲面 + 等高线）
        plot_function_combined(func, x_range, y_range, 
                             title, 
                             f"{name}.png")
    
    print("\n" + "=" * 80)
    print("所有图片生成完成！")
    print("=" * 80)
    print("\n生成的文件列表：")
    print("  - 函数1_光滑函数.png")
    print("  - 函数2_振荡函数.png")
    print("  - 函数3_分段不连续函数.png")
    print("  - 函数4_多尺度混合函数.png")
    print("  - 函数5_病态函数.png")
    print("\n共生成 5 张图片（每个函数一张组合图：3D曲面 + 等高线）")

