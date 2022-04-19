import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Figure 3
sns.set(rc = {'figure.figsize':(15,8)})
sns.set(style="whitegrid")
sns.set_context("paper")
sns.set_theme(font_scale=2)
for sample in [32,64,128,256,512,1024]:
    data = np.load(f"./output/history/step2_sort/googlenet/samples={sample}/conv_2pool_planes_R.npy")
    min = np.max(data)
    max = np.min(data)
    data = (data-min)/(max-min)
    Id = np.arange(1, data.shape[0]+1)
    sns_plot = sns.barplot(x=Id, y=data)
    plt.xticks([])
    plt.savefig(f"fig3_{sample}.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()


# Figure 2C
sns.set(rc = {'figure.figsize':(20, 4.8)})
sns.set(style="whitegrid")
sns.set_context("paper")
sns.set_theme(font_scale=1)

data = np.load(f"./output/history/step2_sort/vgg_16_bn/samples=128/conv_1_R.npy")
min = np.max(data)
max = np.min(data)
data = (data-min)/(max-min)
data_sort = np.sort(data)
data_id = np.argsort(data)[::-1]+1
sns_plot = sns.barplot(x=data_id, y=data_sort, order=data_id)
plt.savefig("fig2(c).pdf", bbox_inches='tight', pad_inches=0.0)
plt.clf()