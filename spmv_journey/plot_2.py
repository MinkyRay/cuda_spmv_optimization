import matplotlib.pyplot as plt
import numpy as np

# 数据准备 (M=400,000 时的稳态数据)
L_widths = [8, 32, 64, 128, 256]
scalar_bandwidth = [182.11, 181.48, 182.38, 151.59, 85.61]
vector_bandwidth = [68.02, 155.82, 182.76, 185.40, 186.33]

plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(L_widths, scalar_bandwidth, 'o-', label='Scalar (Thread-per-Row)', color='#e74c3c', linewidth=2)
plt.plot(L_widths, vector_bandwidth, 's-', label='Vector (Warp-per-Row)', color='#3498db', linewidth=2)

# 硬件理论带宽参考线 (RTX 4050 约 192 GB/s)
plt.axhline(y=192, color='gray', linestyle='--', alpha=0.5, label='Theoretical Peak (192 GB/s)')

# 图表美化
plt.title('SpMV Performance Analysis: Bandwidth vs. Row Width (M=400,000)', fontsize=14)
plt.xlabel('Row Width (L)', fontsize=12)
plt.ylabel('Effective Bandwidth (GB/s)', fontsize=12)
plt.xticks(L_widths)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# 标注关键拐点
plt.annotate('L=8: Scalar dominates', xy=(8, 182), xytext=(20, 190),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
plt.annotate('L=256: Vector dominates', xy=(256, 186), xytext=(150, 160),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
plt.show()