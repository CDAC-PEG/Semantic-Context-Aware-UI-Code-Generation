import os
import numpy as np
from matplotlib import pyplot as plt

if not os.path.exists("..\\Graphs\\"):
    os.mkdir("..\\Graphs\\")

# Double Plateau Histogram Equalization (DPHE)
# Adaptive Histogram Equalization (AHE)
# Histogram Equalization (HE)

def MSE_RMSE_MAE_MAPE():
    plt.figure(figsize=(8, 5))
    Iteration = ['MSE', 'RMSE', 'MAE', 'MAPE']
    ProposedTMBWO = [0.1548,	0.1065,	0.2487,	1.2659]
    ExistingBWO = [0.8589,	0.9586,	0.9865,	3.7412]
    ProposedGWO = [1.2659,	1.4125,	1.4152,	5.8653]
    ExistingPSO = [1.9324,	3.6932,	3.3695,	7.7489]
    ProposedSSO = [2.4716,	4.7451,	4.7523,	9.3568]
    plt.plot(Iteration, ProposedTMBWO, 'H-r', linestyle='-.', markerfacecolor = 'lime')
    plt.plot(Iteration, ExistingBWO, 'h-b', linestyle='-.', markerfacecolor = 'yellow')
    plt.plot(Iteration, ProposedGWO, 'p-y', linestyle='-.', markerfacecolor = 'red')
    plt.plot(Iteration, ExistingPSO, 'd-g', linestyle='-.', markerfacecolor = 'orange')
    plt.plot(Iteration, ProposedSSO, '^-c', linestyle='-.', markerfacecolor = 'magenta')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.xlabel("Metrics", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Values", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed MVE-CLAHE', 'CLAHE', 'DPHE', 'AHE', 'HE'], loc=2, ncol=1)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.tight_layout()
    plt.savefig("..\\Graphs\\MSE_RMSE_MAE_MAPE.png")
    plt.close()
MSE_RMSE_MAE_MAPE()

def UML_Diagram_Grouping():
    Iteration = ['Proposed\nCFM-VCKMeans', 'K-Means', 'PAM', 'MS', 'FCM']
    ProposedTMBWO = [12457, 19658, 24156, 28362, 33471]
    plt.subplots(figsize=(8, 5))
    plt.plot(Iteration, ProposedTMBWO, 'o-b')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.xlabel("Techniques", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("UML Diagram Grouping Time (ms)", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.tight_layout()
    plt.savefig("..\\Graphs\\Diagram_Grouping.png")
UML_Diagram_Grouping()

def SCG_APSSN():
    plt.figure(figsize=(8, 5))
    Iteration = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'NPV']
    ProposedTMBWO = [98.23,	98.45,	98.32,	98.17,	98.62]
    ExistingBWO = [96.35,	96.52,	95.84,	96.47,	95.12]
    ProposedGWO = [94.08,	94.56,	94.84,	93.98,	93.94]
    ExistingPSO = [92.84,	91.32,	91.48,	92.65,	91.54]
    ProposedSSO = [89.23,	88.45,	88.74,	89.62,	88.49]
    plt.plot(Iteration, ProposedTMBWO, marker='H', linestyle='-.', markerfacecolor = 'lime')
    plt.plot(Iteration, ExistingBWO, marker='h', linestyle='-.', markerfacecolor = 'yellow')
    plt.plot(Iteration, ProposedGWO, marker='p', linestyle='-.', markerfacecolor = 'red')
    plt.plot(Iteration, ExistingPSO, marker='d', linestyle='-.', markerfacecolor = 'orange')
    plt.plot(Iteration, ProposedSSO, marker='^', linestyle='-.', markerfacecolor = 'magenta')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.ylim(85, 100)
    plt.title("Performance Evaluation for Source Code Generation", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed TL-LAASTT5', 'AST-T5', 'GPT', 'VAE', 'GNN'], loc=4, ncol = 3)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.tight_layout()
    plt.savefig("..\\Graphs\\SCG_APSSN.png")
    plt.close()
SCG_APSSN()

def OD_APRFMTT():
    plt.figure(figsize=(8, 5))
    Iteration = ['Accuracy', 'PPV', 'Recall', 'F-Measure', 'TPR', 'TNR']
    ProposedTMBWO = [98.23,	98.45,	98.32,	98.17,	98.62,	98.47]
    ExistingBWO = [96.35,	96.52,	95.84,	96.47,	95.12,	95.63]
    ProposedGWO = [94.08,	94.56,	94.84,	93.98,	93.94,	94.65]
    ExistingPSO = [92.84,	91.32,	91.48,	92.65,	91.54,	92.15]
    ProposedSSO = [89.23,	88.45,	88.74,	89.62,	88.49,	89.65]
    plt.plot(Iteration, ProposedTMBWO, 'o-r')
    plt.plot(Iteration, ExistingBWO, 'o-b')
    plt.plot(Iteration, ProposedGWO, 'o-y')
    plt.plot(Iteration, ExistingPSO, 'o-g')
    plt.plot(Iteration, ProposedSSO, 'o-c')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.title("Performance Evaluation for Object Detection", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xlabel("Metrics", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Values (%)", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed WB-YOLO', 'YOLO', 'FR-CNN', 'Mask RCNN', 'R-CNN'], loc=2, ncol=3)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.ylim(88, 102)
    plt.tight_layout()
    plt.savefig("..\\Graphs\\OD_APRFMTT.png")
    plt.close()
OD_APRFMTT()

def SCG_FF():
    ProposedPCNN = [0.9841,	0.9763]
    ExistingCNN = [0.9365,	0.8415]
    ExistingGRU = [0.8927,	0.7635]
    ExistingLSTM = [0.8412,	0.5259]
    ExistingRNN = [0.8036,	0.3512]
    barWidth = 0.15
    br1 = np.arange(len(ProposedPCNN))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    plt.figure(figsize=(8, 5))
    plt.bar(br1, ProposedPCNN, color='#8B7D6B', width=barWidth, edgecolor='antiquewhite', label='Proposed TL-LAASTT5')
    plt.bar(br2, ExistingCNN, color='mediumaquamarine',  width=barWidth, edgecolor='antiquewhite', label='AST-T5')
    plt.bar(br3, ExistingGRU, color='#A2CD5A', width=barWidth, edgecolor='antiquewhite', label='GPT')
    plt.bar(br4, ExistingLSTM, color='#6495ED', width=barWidth, edgecolor='antiquewhite', label='VAE')
    plt.bar(br5, ExistingRNN, color='#8A3324', width=barWidth, edgecolor='antiquewhite', label='GNN')
    plt.title("Performance Evaluation for Source Code Generation", fontweight='bold', fontname="Times New Roman",
              fontsize=14)
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=16)
    plt.ylabel('Values', fontweight='bold', fontname="Times New Roman", fontsize=16)
    plt.xticks([0.3, 1.3], ['FPR', 'FNR'])
    plt.yticks(fontweight='bold', fontsize=16, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=16, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'
    plt.ylim(0, 1.3)
    plt.legend(loc=1, ncol=3)
    plt.tight_layout()
    plt.savefig("..\\Graphs\\SCG_FF.png")
    plt.close()
SCG_FF()

def PFSSOA():
    plt.figure(figsize=(8, 5))
    Iteration = ['Precision', 'F-Score', 'SOA']
    ProposedTMBWO = [98.5678,	98.6521,	92.3659]
    ExistingBWO = [96.4534,	96.3256,	88.5487]
    ProposedGWO = [94.2516,	94.1872,	83.4712]
    ExistingPSO = [92.3697,	92.3698,	77.9351]
    ProposedSSO = [90.2043,	90.2564,	72.5918]
    plt.plot(Iteration, ProposedTMBWO, 'H-r', linestyle='-.', markerfacecolor = 'lime')
    plt.plot(Iteration, ExistingBWO, 'h-b', linestyle='-.', markerfacecolor = 'yellow')
    plt.plot(Iteration, ProposedGWO, 'p-y', linestyle='-.', markerfacecolor = 'red')
    plt.plot(Iteration, ExistingPSO, 'd-g', linestyle='-.', markerfacecolor = 'orange')
    plt.plot(Iteration, ProposedSSO, '^-c', linestyle='-.', markerfacecolor = 'magenta')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.xlabel("Metrics", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Values (%)", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.legend(['Proposed SXLNet-LSSMGDAN', 'GAN', 'GPT', 'VAE', 'LDM'], loc=3, ncol=1)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.tight_layout()
    plt.savefig("..\\Graphs\\PFSSOA.png")
    plt.close()
# PFSSOA()

def Entropy_CS():
    ProposedTMBWO = [0.8756,	0.9865]
    ExistingBWO = [0.7326,	0.9471]
    ProposedGWO = [0.6147,	0.9023]
    ExistingPSO = [0.5623,	0.8647]
    ProposedSSO = [0.4157,	0.8136]
    barWidth = 0.17
    br1 = np.arange(len(ProposedTMBWO))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    plt.figure(figsize=(8, 5))
    plt.bar(br1, ProposedTMBWO, color='#6495ED', hatch='\\\\', width=barWidth, edgecolor='pink', label='Proposed Grad-GFCAM')
    plt.bar(br2, ExistingBWO, color='salmon', hatch='\\\\', width=barWidth, edgecolor='pink', label='Grad-CAM')
    plt.bar(br3, ProposedGWO, color='#CD950C', hatch='\\\\', width=barWidth, edgecolor='pink', label='Score-CAM')
    plt.bar(br4, ExistingPSO, color='plum', hatch='\\\\', width=barWidth, edgecolor='pink', label='Ablation-CAM')
    plt.bar(br5, ProposedSSO, color='#3D9140', hatch='\\\\', width=barWidth, edgecolor='pink', label='Eigen-CAM')
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([0.35, 1.35], ['Entropy', 'Cosine Similarity'])
    plt.ylim(0, 1.3)
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.legend(loc=2, ncol=2)
    plt.tight_layout()
    plt.savefig("..\\Graphs\\Entropy_CS.png")
    plt.close()
# Entropy_CS()

def FID():
    Iteration = ['Proposed\nSXLNet-LSSMGDAN', 'GAN', 'GPT', 'VAE', 'LDM']
    ProposedTMBWO = [10.2356, 19.7481, 26.8472, 30.2654, 36.9523]
    plt.subplots(figsize=(8, 5))
    plt.plot(Iteration, ProposedTMBWO, 'd-r', linestyle='--')
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.xlabel("Techniques", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("FID", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.tight_layout()
    plt.savefig("..\\Graphs\\FID.png")
# FID()

def Computational_Time():
    courses = ['Proposed SAC-UNet', 'U-Net', 'ResNet', 'DenseNet', 'EfficientNet']
    values = [3258, 5412, 7325, 9057, 11241]
    plt.subplots(figsize=(8, 5))
    colors = ['#43CD80', '#555555', '#87CEEB', '#CD853F', '#EE82EE']
    plt.bar(courses, values, color=colors, width=0.35)
    plt.xlabel("Techniques", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel("Computational Time (ms)", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.yticks(fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks(fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.tight_layout()
    plt.savefig("..\\Graphs\\Computational_Time.png")
# Computational_Time()

def KLD_DBI():
    ProposedGAN = [0.5784,	0.5124]
    ExistingGAN = [0.5123,	0.4765]
    ExistingGPT = [0.4856,	0.4286]
    ExistingVAE = [0.4238,	0.3869]
    ExistingLDM = [0.3787,	0.3367]
    barWidth = 0.17
    br1 = np.arange(len(ProposedGAN))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    plt.figure(figsize=(8, 5))
    plt.bar(br1, ProposedGAN, hatch='--', color='#FFD700', edgecolor='antiquewhite', width=barWidth,  label='Proposed SAC-UNet')
    plt.bar(br2, ExistingGAN, hatch='--', color='#68228B', edgecolor='antiquewhite', width=barWidth, label='U-Net')
    plt.bar(br3, ExistingGPT, hatch='--', color='#EE7600', edgecolor='antiquewhite', width=barWidth,  label='ResNet')
    plt.bar(br4, ExistingVAE, hatch='--', color='#00CED1', edgecolor='antiquewhite', width=barWidth,  label='DenseNet')
    plt.bar(br5, ExistingLDM, hatch='--', color='#FFC1C1', edgecolor='antiquewhite', width=barWidth,  label='EfficientNet')
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([0.35, 1.35], ['KLD',	'Davies–Bouldin Index'])
    plt.yticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.xticks(fontweight='bold', fontsize=14, fontname="Times New Roman")
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.legend(loc=1, ncol=3)
    plt.ylim(0.1, 0.62)
    plt.tight_layout()
    plt.savefig("..\\Graphs\\KLD_DBI.png")
    plt.close()
# KLD_DBI()