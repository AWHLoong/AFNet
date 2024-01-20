Overview
基于双模态自适应融合的腋窝淋巴结状态分类方法
(State Classification of Axillary Lymph Nodes via Dual-modal Adaptive Fusion)

1.System requirements:

- Hardware Requirements
At least NVIDIA GTX 2080Ti

- OS Requirements

This package is supported for windows. The package has been tested on the following systems:
windows: Windows 10 专业版

- Software Prerequisites

torch 1.8.1+cu111
torchvision 0.9.1+cu111
CUDA 10.1
python 3.8
numpy 1.22.3
scikit-learn 1.0.2
matplotlib 3.5.1

2.Installation guide:

It is recommended to install the environment in the windows system.
First install Anconda3.
Then install CUDA 10.x and cudnn.
Finall intall these dependent python software library.
The installation is estimated to take 1 hour, depending on the network environment

3.Demo:
- 打开test.py测试示例
- 通过 plot_class_preds 通过参数设定加载数据集路径（bmode_images_dir、swe_images_dir分别代表B-mode图像和SWE图像的路径)
- 运行test.py直接对dataset数据集进行预测，最后保存结果文件在utils目录下（文件名为：test_result_acc_auc_xxx.jpg）
- 通过预测结果文件可以看到各个示例预测的概率和病理结果的对比，并计算最终的acc和auc


Note:
B-mode: B-mode US
SWE: shear wave elastography