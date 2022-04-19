import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
sns.set_context("paper")

# vgg16
plt.ylim(0, 64) 
datanp = np.array([], dtype=int)
for i in [8,16,32,64,128,256,512,1024]:
    temp = np.genfromtxt(f"./output/history/step0_appropriateK/vgg_16_bn/conv_1/samples={i}.csv", delimiter=",", dtype=int) # vgg16
    if datanp.size == 0:
        datanp = temp
    else:
        datanp = np.concatenate((datanp, temp), axis=0)
data = pd.DataFrame(datanp, columns = ['optimal K', '',])
sns_plot = sns.boxplot(x="", y="optimal K", data=data)

bestK = data.mode()["optimal K"][0]
sns_plot.axhline(bestK, color='coral', linestyle=':', linewidth=3)
bbox = dict(boxstyle="round", fc="0.8")
plt.annotate('K='+str(bestK), color='coral', xy=(0, bestK), xytext=(-70, -5), textcoords='offset points', bbox=bbox, fontsize='large', fontweight='bold')
plt.savefig("fig4(a).pdf", bbox_inches='tight', pad_inches=0.0) # vgg16
plt.clf()

# densenet
plt.ylim(0, 168) 
datanp = np.array([], dtype=int)
for i in [8,16,32,64,128,256,512,1024]:
    temp = np.genfromtxt(f"output/history/step0_appropriateK/densenet/conv_14/samples={i}.csv", delimiter=",", dtype=int) # densenet
    if datanp.size == 0:
        datanp = temp
    else:
        datanp = np.concatenate((datanp, temp), axis=0)
data = pd.DataFrame(datanp, columns = ['optimal K', '',])
sns_plot = sns.boxplot(x="", y="optimal K", data=data)

bestK = data.mode()["optimal K"][0]
sns_plot.axhline(bestK, color='coral', linestyle=':', linewidth=3)
bbox = dict(boxstyle="round", fc="0.8")
plt.annotate('K='+str(bestK), color='coral', xy=(0, bestK), xytext=(-70, -5), textcoords='offset points', bbox=bbox, fontsize='large', fontweight='bold')
plt.savefig("fig4(b).pdf", bbox_inches='tight', pad_inches=0.0) # densenet
plt.clf()

# resnet56
plt.ylim(0, 32) 
datanp = np.array([], dtype=int)
for i in [8,16,32,64,128,256,512,1024]:
    temp = np.genfromtxt(f"output/history/step0_appropriateK/resnet_56/conv_25/samples={i}.csv", delimiter=",", dtype=int)  # resnet56
    if datanp.size == 0:
        datanp = temp
    else:
        datanp = np.concatenate((datanp, temp), axis=0)
data = pd.DataFrame(datanp, columns = ['optimal K', '',])
sns_plot = sns.boxplot(x="", y="optimal K", data=data)

bestK = int(data["optimal K"].median())
sns_plot.axhline(bestK, color='coral', linestyle=':', linewidth=3)
bbox = dict(boxstyle="round", fc="0.8")
plt.annotate('K='+str(bestK), color='coral', xy=(0, bestK), xytext=(-70, -5), textcoords='offset points', bbox=bbox, fontsize='large', fontweight='bold')
plt.savefig("fig4(c).pdf", bbox_inches='tight', pad_inches=0.0) # resnet56
plt.clf()