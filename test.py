import math
import os
import time

import numpy as np
import torch
from PIL import Image
from pandas import DataFrame
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_curve, auc
from torch.utils.data import DataLoader

from models.RMT import RMT_S_pretrain, RMT_L_pretrain
from models.models_list import vit_s, swin_s
from models.overlock import OverLock_B_pretrain
from models.starnet import starnet_s4_pretrain

from dataset import FLDataset


class ReshapeTransform:
    def __init__(self, model):
        pass

    def __call__(self, x):
        result = x[:, 1:, :].reshape(x.size(0), 14, 14, x.size(2))
        result = result.permute(0, 3, 1, 2)
        return result


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0), self.height, self.width, x.size(3))
        result = result.permute(0, 3, 1, 2)
        return result


def get_layer(model, model_name):
    if model_name == 'resnet18':
        return [model.resnet.layer4[1].conv2]
    if model_name == 'resnet50':
        return [model.resnet.layer4[2].conv3]
    if model_name == 'resnet101':
        return [model.resnet.layer4[2].conv3]
    if model_name == 'swinT':
        return [model.net.norm]
    if model_name == 'vit':
        return [model.net.blocks[-1].norm1]
    if model_name == 'vig':
        return [model.backbone[-1][0].fc1[1]]


def infer_cls():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    # -------------------------------------------------------
    size = 224
    device = torch.device('cuda')
    dataset_name = 'TB' # SIIM, MIMIC, TB, COVID
    label_num = 2  # 2 3 2 3

    model = OverLock_B_pretrain(label_num)

    # -------------------------------------------------------

    base = "output/exp_overlockb_e4/"
    model_path = base + "0.8078_26_checkpoint.pth"
    # -------------------------------------------------------
    # output_dir = base + "Attention map/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    model.eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    dataset_test = FLDataset('汇总数据-统一为6张', 'test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    pred_list = []
    label_list = []
    pred_score_list = []
    gt = []
    pd = []
    img_name_list = []
    dur_list = []
    for img, label, img_path in dataloader_test:
        img = img.to(device)
        label = label.to(device)
        img_name_list.append(img_path[0][0])
        # -------------------------------------------------------
        # pred
        # -------------------------------------------------------
        t = time.time()
        print(img.shape)
        output = model(img)
        dur = time.time() - t
        print(dur)
        dur_list.append(dur)
        _, pred = torch.max(output, 1)
        pred_score = torch.nn.Softmax(dim=1)(output)
        # -------------------------------------------------------
        # attention map
        # -------------------------------------------------------
        # target_layer = get_layer(model, model_name)
        # if model_name == 'swinT':
        #     cam = GradCAMPlusPlus(model=model, target_layers=target_layer,
        #                           reshape_transform=ResizeTransform(im_h=size, im_w=size))
        # elif model_name == 'vit':
        #     cam = GradCAMPlusPlus(model=model, target_layers=target_layer,
        #                           reshape_transform=ReshapeTransform(model))
        # else:
        #     cam = GradCAMPlusPlus(model=model, target_layers=target_layer)
        # targets = [ClassifierOutputTarget(label.item())]
        # grayscale_cam = cam(input_tensor=img, targets=targets)
        # grayscale_cam = grayscale_cam[0, :]
        #
        # img = Image.open(img_path[0]).resize((size, size))
        # img_float_np = np.float32(img) / 255.
        # img_float_np = np.expand_dims(img_float_np, axis=-1).repeat(3, axis=-1)
        #
        # cam_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)
        # cam_image = Image.fromarray(cam_image)
        # cam_image.save(output_dir + img_name)
        # -------------------------------------------------------
        # append to list
        # -------------------------------------------------------
        pred_score_list.append(pred_score[0][1].cpu().detach().numpy().tolist())
        pred_list.append(pred.cpu().detach().numpy().tolist())
        label_list.append(label.cpu().detach().numpy().tolist())
        gt.extend(label.cpu().detach().numpy())
        pd.extend(output.cpu().detach().numpy())

    pred_list = [b for a in pred_list for b in a]
    label_list = [b for a in label_list for b in a]
    # -------------------------------------------------------
    # print metrics
    # -------------------------------------------------------
    dur = np.array(dur_list).mean()
    print('Avg Time   : {:.4f}'.format(dur))

    acc = accuracy_score(label_list, pred_list)
    print('Accuracy   : {:.4f}'.format(acc))

    precision = precision_score(label_list, pred_list, average='weighted')
    print('Precision  : {:.4f}'.format(precision))

    recall = recall_score(label_list, pred_list, average='weighted')
    print('Recall     : {:.4f}'.format(recall))

    f1 = f1_score(label_list, pred_list, average='weighted')
    print('F1 score   : {:.4f}'.format(f1))

    cm = confusion_matrix(label_list, pred_list)
    print(cm)

    if dataset_name == 'SIIM' or dataset_name == 'TB':
        auc_score = metrics.roc_auc_score(label_list, pred_score_list)
        print('AUC        : {:.4f}'.format(auc_score))
    else:
        gt = np.array(gt)
        pd = np.array(pd)
        gt = label_binarize(np.array(gt), classes=[0, 1, 2])
        fpr = dict()
        tpr = dict()
        roc_auc = []
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(gt[:, i], pd[:, i])
            roc_auc.append(auc(fpr[i], tpr[i]))
        aucavg = np.mean(roc_auc)
        print("AUC: {}".format(roc_auc))
        print("Avg AUC: {}".format(aucavg))


    # hospital_name = [str(img_path).split('imageCutErWei')[0].split('/')[1] for img_path in img_name_list]
    # patient_name = [str(img_path).split('imageCutErWei')[0].split('/')[3] for img_path in img_name_list]
    # class_name = [str(img_path).split('imageCutErWei')[0].split('/')[2] for img_path in img_name_list]
    # gt_list = gt
    # pred_list = pred_list
    # pred_score_list_0 = [1-p for p in pred_score_list]
    # pred_score_list_1 = pred_score_list
    #
    # names = ["医院", "姓名", "疾病类型", "标签", "预测", "预测分数-滤泡", "预测分数-腺瘤"]
    #
    # # 创建 DataFrame
    # df = DataFrame({
    #     "医院": hospital_name,
    #     "姓名": patient_name,
    #     "疾病类型": class_name,
    #     "标签": gt_list,
    #     "预测": pred_list,
    #     "预测分数-滤泡": pred_score_list_0,
    #     "预测分数-腺瘤": pred_score_list_1
    # })
    #
    # # 如果你需要保存到 Excel 或 CSV：
    # df.to_excel(base + "预测结果_train.xlsx", index=False)  # 保存为 Excel 文件


if __name__ == '__main__':
    infer_cls()
