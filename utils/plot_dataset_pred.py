import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
def plot_class_preds(net,
                     bmode_images_dir: str,
                     swe_images_dir: str,
                     transform,
                     num_plot: int = 5,
                     device="cpu"):
    # 自动进行调用获取类别字典
    index_to_label = dict({0: "non-metastasis", 1: "metastasis"})
    label_to_index = dict((name, i) for (i, name) in index_to_label.items())
    # 判断是否存在测试集的图片
    if not os.path.exists(bmode_images_dir):
        print("not found {} path, ignore add figure.".format(bmode_images_dir))
        return None
    if not os.path.exists(swe_images_dir):
        print("not found {} path, ignore add figure.".format(swe_images_dir))
        return None
    # 获取标签，直接从图片中获取就可以了
    all_testImg_path_bmode = glob.glob(bmode_images_dir + "/*.jpg")
    all_testImg_path_bmode.sort(key=lambda x: (x.split('\\')[1].split('.')[0]))

    all_testImg_path_swe = glob.glob(swe_images_dir + "/*.jpg")
    all_testImg_path_swe.sort(key=lambda x: (x.split('\\')[1].split('.')[0]))


    label = []  #
    label_info = []
    for bmode_path, swe_path in zip(all_testImg_path_bmode, all_testImg_path_swe):
        label.append(label_to_index.get(bmode_path.split("\\")[1].split(".")[0]))
        class_name = bmode_path.split("\\")[1].split(".")[0]
        label_info.append([bmode_path, swe_path, class_name])
    if len(label_info) == 0:
        return None
    if num_plot == None:
        pass
    else:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images_bmode = []
    images_swe = []
    labels = []
    image_name_list = []

    for img_path_bmode, img_path_swe, class_name in label_info:
        image_name_list.append(img_path_bmode.split("\\")[1].split(".jpg")[0])
        #print(img_name)
        # read img
        img_bmode = Image.open(img_path_bmode).convert("RGB")
        img_swe = Image.open(img_path_swe).convert("RGB")
        label_index = int(label_to_index[class_name])
        # preprocessing
        img_bmode = transform(img_bmode)
        img_swe = transform(img_swe)
        images_bmode.append(img_bmode)
        images_swe.append(img_swe)
        labels.append(label_index)
    # batching images
    images_bmode = torch.stack(images_bmode, dim=0).to(device)
    images_swe = torch.stack(images_swe, dim=0).to(device)
    # 进入预测模式
    net.eval()

    with torch.no_grad():

        output = net(images_bmode, images_swe)
        outputs_sf = torch.softmax(output, dim=1)
        # 得到最好的极端值，根据截断值就散预测的值
        preds = list(map(lambda x: 1 if x >= 0.5 else 0, outputs_sf[:,1]))
        probs = outputs_sf.cpu().numpy()[:, 1]


    auc_compute = roc_auc_score(np.array(labels), outputs_sf.cpu().numpy()[:, 1])
    # width, height
    total_column = 6
    total_row = num_imgs // total_column if num_imgs % total_column == 0 else num_imgs // total_column + 1
    # 计算 acc
    acc = round((sum(np.array(preds) == np.array(labels)) / len(labels)), 3)
    print(f"ACC:{acc},AUC:{round(auc_compute, 3)},plot_testImg_pred,img_len:{num_imgs * 2}")
    fig = plt.figure(figsize=(total_column * 3, total_row * 8), dpi=100)
    # 保存整个图
    for i in range(num_imgs):
        ax = fig.add_subplot(total_row * 2, total_column, 2 * i + 1, xticks=[], yticks=[])
        # CHW -> HWC
        npimg_bmode = images_bmode[i].cpu().numpy().transpose(1, 2, 0)
        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg_bmode = (npimg_bmode * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

        plt.imshow(npimg_bmode.astype('uint8'))

        '''swe 图像'''
        ax = fig.add_subplot(total_row * 2, total_column, 2 * i + 2, xticks=[], yticks=[])

        npimg_swe = images_swe[i].cpu().numpy().transpose(1, 2, 0)
        npimg_swe = (npimg_swe * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

        plt.imshow(npimg_swe.astype('uint8'))

        title = "pred:{}\n prob {:.5f}\n(label: {})\n{}".format(
            index_to_label[preds[i]],  # predict class
            probs[i],  # predict probability
            index_to_label[labels[i]],
            image_name_list[i]   # true class
        )


        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))
    # 先保存错误的子图吧
    for i in range(num_imgs):
        # CHW -> HWC
        npimg_bmode = images_bmode[i].cpu().numpy().transpose(1, 2, 0)
        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg_bmode = (npimg_bmode * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        '''swe 图像'''
        npimg_swe = images_swe[i].cpu().numpy().transpose(1, 2, 0)
        npimg_swe = (npimg_swe * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        fig1 = plt.figure(figsize=(2 * 3, 1 * 3.5), dpi=100)
        # bmode 图像
        fig1.add_subplot(1, 2, 1, xticks=[], yticks=[])
        plt.imshow(npimg_bmode.astype('uint8'))

        '''swe 图像'''
        ax2 = fig1.add_subplot(1, 2, 2, xticks=[], yticks=[])

        plt.imshow(npimg_swe.astype('uint8'))

    return fig, acc,auc_compute