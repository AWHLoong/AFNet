from torchvision import transforms
import torch
from utils import creatDataset,plot_dataset_pred

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")



data_transform = {
    "train": transforms.Compose([transforms.Resize(224),
                                 transforms.RandomHorizontalFlip(),  # 随机翻转，数据增强
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



def main():
    test_loader = creatDataset.generate_ds(dataSet_root_dir="../testDataset_part/dataset/",
                                              transforms=data_transform,
                                              batch_size=5,
                                              isShuffle=False)
    model = torch.jit.load("../weights/AFNeta.pth", map_location=device)


    model.eval()

    # for test_data in tqdm(test_loader):
    #     (test_images_bmode, test_images_swe), test_labels = test_data
    #     test_outputs = model(test_images_bmode.to(device), test_images_swe.to(device))
    #     test_outputs_sf = torch.softmax(test_outputs, dim=1)
    # print(test_outputs_sf)

    test_fig, test_acc, test_auc = plot_dataset_pred.plot_class_preds(net=model,
                                                                      bmode_images_dir="../testDataset_part/dataset/Bmode/test",
                                                                      swe_images_dir="../testDataset_part/dataset/Swe/test",
                                                                      transform=data_transform["val"],
                                                                      num_plot=20,
                                                                      device=device)

    test_fig.savefig( "./test_result_acc_auc " + str(round(test_acc, 3)) + "_" + str(round(test_auc, 3)) + "_.jpg")

if __name__ == '__main__':
    main()
