import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 整理实验数据
'''
data = {
    'Row_Len': [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 
                32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128],
    'SubWarp': [2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 
                2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32],
    'GFLOPS': [25.88, 26.89, 12.40, 16.59, 9.93, 24.64, 34.99, 29.16, 33.28, 19.31, 
               51.15, 71.35, 65.47, 62.25, 38.16, 45.37, 65.24, 88.32, 82.35, 72.38, 
               46.14, 64.20, 90.71, 107.81, 105.75, 46.57, 65.71, 85.30, 119.31, 132.70]
}
'''
data = {
    'Row_Len': [4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 
                32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128],
    'SubWarp': [2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 
                2, 4, 8, 16, 32, 2, 4, 8, 16, 32, 2, 4, 8, 16, 32],
    'GFLOPS': [16.64, 16.40, 12.34, 15.69, 9.81, 36.47, 26.58, 24.26, 24.19, 19.68, 
               50.98, 71.76, 49.72, 62.32, 33.48, 46.16, 63.04, 74.76, 73.49, 64.28, 
               47.13, 65.39, 85.59, 110.57, 96.38, 47.60, 67.14, 86.96, 117.81, 131.48]
}

df = pd.DataFrame(data)

# 绘制折线图：观察不同行长下 SWS 的敏感度
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
sns.lineplot(data=df, x='SubWarp', y='GFLOPS', hue='Row_Len', marker='o', palette='viridis')

plt.title('SpMV Performance: SubWarp Size vs. GFLOPS (Varying Row Lengths)', fontsize=15)
plt.xlabel('SubWarp Size (SWS)', fontsize=12)
plt.ylabel('Throughput (GFLOPS)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks([2, 4, 8, 16, 32], [2, 4, 8, 16, 32])
plt.legend(title='Row Length', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 绘制热力图：寻找最优配置区域
pivot_df = df.pivot(index="Row_Len", columns="SubWarp", values="GFLOPS")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Heatmap: Best SWS for Different Row Lengths', fontsize=15)
plt.show()