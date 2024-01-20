import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from matplotlib import pyplot as plt


index_to_label = dict({0: "non-metastasis", 1: "metastasis"})
label_to_index = dict((name, i) for (i, name) in index_to_label.items())


def plot_data_loader_image(data_loader, batch_size=8, plotBatchOfNums=4, datasetType="bmode", unNormal=1, save_dir=" "):


    #clear_and_create_dir(save_dir)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei黑体  FangSong仿宋
    plt.rcParams['axes.unicode_minus'] = False
    # batch_size = data_loader.batch_size
    plot_num = min(batch_size, plotBatchOfNums)
    for data in data_loader:
        plt.figure(figsize=(20, 8))
        (bmode_img, swe_img), labels = data
        if len(labels) < plot_num:
            plot_num = len(labels)

        for i in range(plot_num):
            b_img = bmode_img[i].numpy().transpose(1, 2, 0)
            s_img = swe_img[i].numpy().transpose(1, 2, 0)
            label = labels[i].numpy()
            if unNormal == 1:
                # 反Normalize操作
                b_img = (b_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                s_img = (s_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            elif unNormal == 2:
                b_img = (b_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                s_img = (s_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

            else:
                b_img = (b_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                s_img = (s_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255

            plt.subplot(2, plot_num, i + 1)

            plt.imshow(b_img.astype('uint8'))
            plt.subplot(2, plot_num, i + 1 + plot_num)
            plt.xlabel(index_to_label.get(int(label)))
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(s_img.astype('uint8'))
        plt.show()

class MyDataSet_Bmode_swe(Dataset):
    """自定义数据集"""

    def __init__(self, bmode_images_path: list, swe_images_path: list, transform=None):
        self.bmode_images_path = bmode_images_path
        self.swe_images_path = swe_images_path
        # self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.bmode_images_path)

    def __getitem__(self, item):
        img_path = self.bmode_images_path[item]
        bmode_img = Image.open(self.bmode_images_path[item])
        swe_img = Image.open(self.swe_images_path[item])
        # RGB为彩色图片，L为灰度图片
        if (bmode_img.mode != 'RGB') | (swe_img.mode != 'RGB'):
            raise ValueError("image: {} isn't RGB mode.".format(self.bmode_images_path[item]))
        # label = self.images_class[item]

        label = img_path.split("\\")[-1].split(".")[0]

        label = 1 if label == "metastasis" else 0

        if self.transform is not None:
            bmode_img = self.transform(bmode_img)
            swe_img = self.transform(swe_img)

        return [bmode_img, swe_img], label

class my_Dateset():
    def __init__(self, datalist, transforms):
        self.test_Bmode = datalist[0]
        self.test_Swe = datalist[1]
        self.test_img_name = datalist[2]
        self.data_transform = transforms
    # 根据参数加载数据集
    def load(self, BATCH_SIZE=16, Isshuffle=False):
        test_data = MyDataSet_Bmode_swe(self.test_Bmode, self.test_Swe, transform=self.data_transform["val"])
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)

        return test_loader

def generate_ds(dataSet_root_dir="",transforms="", batch_size=8, isShuffle=False):

    test_Bmode = glob.glob(dataSet_root_dir + "Bmode/test/" + '*.jpg')  # 得到所有Bmode 图像的路径
    test_Swe = glob.glob(dataSet_root_dir + "Swe/test/" + '*.jpg')


    test_img_name = [path.split('\\')[1][:-4] for path in test_Bmode]

    dataset = my_Dateset([test_Bmode,test_Swe,test_img_name],transforms=transforms)  # 创建数据集的对象
    '''==========================================训练/测试数据集 显示 开始====================================================='''

    # 加载数据集 [bmode,swe],label -batch
    test_loader = dataset.load(BATCH_SIZE=batch_size, Isshuffle=isShuffle)

    '''==========================================训练/测试数据集 显示 结束====================================================='''
    return test_loader

if __name__ == "__main__":

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



    test_loader = generate_ds(dataSet_root_dir="../testDataset_part/dataset/",
                              transforms=data_transform,
                              batch_size=1,
                              isShuffle=False)


    print(f"train_loader :{len(test_loader)}")

    plot_data_loader_image(test_loader, batch_size=1, plotBatchOfNums=4, datasetType="bmode_swe", unNormal=2,
                           save_dir="./data_loader/" +"a" + "/test/")