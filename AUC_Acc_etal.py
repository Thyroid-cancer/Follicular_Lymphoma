# -*- coding: utf-8 -*-


import numpy as np
import random
import os
import torch
import pandas as pd
from sklearn.metrics import auc,roc_auc_score,roc_curve,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import norm
import random


def delong_roc_test(y_true, prob_1, prob_2):
    """
    """
    def compute_auc_variance(y_true, prob):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_value = auc(fpr, tpr)
        n1 = sum(y_true == 1)
        n2 = sum(y_true == 0)
        v = (auc_value * (1 - auc_value) + (n1 - 1) * (auc_value / (2 - auc_value))**2 +
             (n2 - 1) * ((1 - auc_value) / (1 + auc_value))**2) / (n1 * n2)
        return auc_value, v

    auc1, var1 = compute_auc_variance(y_true, prob_1)
    auc2, var2 = compute_auc_variance(y_true, prob_2)
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p_value = 2 * (1 - norm.cdf(abs(z)))  # 双侧检验
    return p_value

def extractor_excel_RFS(xlsx):
    Score = np.array(xlsx['Score1'])
    Label = np.array(xlsx['Label'])
    return Score,Label


if __name__ == '__main__':
    ### 提取临床指标特征
    xlsx_path = 'C:/Users/data.xlsx'
    xlsx_train = pd.read_excel(xlsx_path, sheet_name='Train')
    xlsx_val = pd.read_excel(xlsx_path, sheet_name='Val')
    xlsx_test = pd.read_excel(xlsx_path, sheet_name='Test')
    
    Score,label = extractor_excel_RFS(xlsx_val)
    
    prob = np.column_stack((1-Score, Score))
    pred = np.argmax(prob, axis = 1)
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    
    # 计算各类指标
    auc=roc_auc_score(label,prob[:,1]) #计算auc
    fpr,tprr,thresholds=roc_curve(label,prob[:,1])  #计算fpr,tpr,thresholds
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred)
    spec = tn / (tn + fp)  # 特异性（真负例率）
    sen = tp / (tp + fn)  # 召回率（真正例率）
    ppv = tp / (tp + fp)  # 阳性预测值
    npv = tn / (tn + fn)  # 阴性预测值






























