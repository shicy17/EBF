import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams, cm

# score1
top_data = pd.DataFrame({
    10: [0.663941, 0.861688, 0.92426, 0.958149],  # 10000
    15: [0.576094, 0.803668, 0.884406, 0.929746],  # 15000
    20: [0.507085, 0.754591, 0.848888, 0.902847],  # 20000
    25: [0.447836, 0.708052, 0.814838, 0.876412],  # 25000
    30: [0.398465, 0.665832, 0.784515, 0.851928],  # 30000
    35: [0.358558, 0.628331, 0.756513, 0.829388],  # 35000
    40: [0.320536, 0.590869, 0.729251, 0.806757]  # 40000
})
bottom_data = pd.DataFrame({
    10: [0.0953634, 0.236938, 0.385933, 0.506601],
    15: [0.0571708, 0.155015, 0.27395, 0.38295],
    20: [0.0423879, 0.114383, 0.210565, 0.306299],
    25: [0.027195, 0.0825676, 0.16253, 0.246794],
    30: [0.0229645, 0.0666155, 0.133288, 0.207247],
    35: [0.0258338, 0.0609939, 0.117575, 0.183343],
    40: [0.0202437, 0.0494369, 0.0983569, 0.157124]
})

import sys
# flag = sys.argv[1]
# if flag == 'dri':
#dri auc
# top_data = pd.DataFrame({
#     1: [0.86,0.88,0.91,0.92],  # BAF STCF QMLPF EBF
#     3: [0.85,0.89,0.91,0.92],  
#     5: [0.83,0.89,0.90,0.92]  
# })
# if flag == 'campus':
#campus auc
# bottom_data = pd.DataFrame({
#     1: [0.78,0.81,0.84,0.87],  # BAF STCF QMLPF EBF
#     3: [0.77,0.81,0.83,0.87],  
#     5: [0.76,0.80,0.83,0.86]  
# })

# bottom_data = pd.DataFrame({
#     1: [0.78,0.81,0.832,0.855],  # BAF STCF EDformer EBF
#     3: [0.77,0.81,0.830,0.855],  
#     5: [0.76,0.80,0.828,0.851]  
# })

# bottom_data = pd.DataFrame({
#     1: [0.78,0.81,0.832,0.87],  # BAF STCF EDformer EBF tau = 64 ms
#     3: [0.77,0.81,0.830,0.866],  
#     5: [0.76,0.80,0.828,0.864]  
# })

# bottom_data = pd.DataFrame({
#     1: [0.78,0.81,0.832,0.886],  # BAF STCF EDformer EBF tau=32 ms # 3M
#     3: [0.77,0.81,0.830,0.882],  
#     5: [0.76,0.80,0.828,0.880]  
# })

bottom_data = pd.DataFrame({
    1: [0.78,0.81,0.845,0.881],  # BAF STCF EDformer EBF tau=32 ms # all campus data
    3: [0.77,0.81,0.845,0.879],  
    5: [0.76,0.80,0.844,0.877]  
})


# top_data = pd.DataFrame({
#     1: [0.88,0.71,0.92,0.93],  # BAF STCF QMLPF EBF
#     3: [0.88,0.73,0.92,0.92],  
#     5: [0.87,0.76,0.92,0.92]  
# })


rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 20

# 设置柱状图颜色
colors_top = ['blue', 'green', 'red', 'cyan']  # 顶部柱子的颜色
colors_bottom = ['lightblue', 'lightgreen', 'lightcoral', 'lightcyan']  # 底部柱子的颜色

# colors_top = cm.Set2(np.linspace(0, 1, len(top_data)))
# colors_bottom = cm.Set1(np.linspace(0, 0.5, len(bottom_data)))

# 绘制图表 0, 0.5 set2
# 不去频闪
# colors_top = cm.Set1(np.linspace(0, 1, len(top_data)))
colors_bottom = cm.Set3(np.linspace(0.5, 1, len(bottom_data)))
# 去频闪
# colors_top = cm.Set1(np.linspace(0, 0.5, len(top_data)))
# colors_bottom = cm.Set3(np.linspace(0.5, 1, len(bottom_data)))

bar_width = 0.2  # 柱子的宽度
plt.rcParams.update({'font.size': 24})
index = np.arange(len(top_data.columns))  # x轴的位置

fig, ax = plt.subplots(figsize=(12, 8))
filters = ['BAF','STCF','EDformer','EBF']
# filters = ['DA','DP','AP','DAP']

x = [1,3,5]
marker_list = ['o', 'v', 
               '^', 'D', 
               '>', 's', 
               'p',  
               'h', 
              ]
color_list = ['k', 'deepskyblue',
             'coral', 'r',
              
             'lightsteelblue', 'navy',
             'blueviolet', 'pink']

# linetype = ['c-','s-','m.','*-']
for i in range(len(top_data)):
    # ax.plot(x,top_data.iloc[i],c = color_list[i], marker = marker_list[i], ms=10,ls='--',linewidth=2, label='D ' + filters[i])
    # ax.plot(x,bottom_data.iloc[i],c = color_list[i], marker = marker_list[i], ms=10, linewidth=2, label='C ' + filters[i])
    ax.plot(x,bottom_data.iloc[i],c = color_list[i], marker = marker_list[i], ms=10, linewidth=2, label=filters[i]) # edformer
    
    # ax.bar(index + i * bar_width, top_data.iloc[i] * 100, bar_width, color=colors_top[i], label=filters[i] + 'on Driving')
    # # (\u03bc={i+1})
    # # \u03BC=
    # ax.bar(index + i * bar_width, bottom_data.iloc[i] * 100, bar_width, color=colors_bottom[i],
    #        label=filters[i] + 'on Campus')

ax.set_xlabel('Noise Level (Hz/pixel)', fontsize=24, fontweight='bold')
ax.set_ylabel('AUC', fontsize=24, fontweight='bold')
ax.set_title('')

# ax.set_xticks(index + bar_width / 2 * (len(top_data) - 1))
# ax.set_xticks(index+bar_width*2)
# ax.set_xticklabels(top_data.columns)

y_values = np.arange(0, 1, 0.1)  # 假设百分比数据，每10%画一条线
for y in y_values:
    ax.axhline(y, color='lightgray', linewidth=1, linestyle='-', zorder=0)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, fontsize=12)
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True, edgecolor='black', fontsize=20,
                   ncol=len(bottom_data))

legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1)
plt.ylim(0.75,0.9)
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('campusallauc0219.pdf') 
    ##score1
plt.show()
