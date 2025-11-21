import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, compounds_bert, proteins_bert, labels, morgan_fp, avalon_fp, maccs_fp, protein_features = data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            compounds_bert = compounds_bert.to(DEVICE)
            proteins_bert = proteins_bert.to(DEVICE)
            labels = labels.to(DEVICE)
            morgan_fp = morgan_fp.to(DEVICE)
            avalon_fp = avalon_fp.to(DEVICE)
            maccs_fp = maccs_fp.to(DEVICE)
            protein_features = protein_features.to(DEVICE)

            if isinstance(MODEL, list):
                final_pred = torch.zeros(labels.size(0), 2).to(DEVICE)
                for i in range(len(MODEL)):
                    outputs = MODEL[i](compounds, proteins, morgan_fp, avalon_fp, maccs_fp, protein_features, compounds_bert, proteins_bert)
                    final_pred = final_pred + outputs[0]  # 只取final_pred
                predicted_scores = final_pred / FOLD_NUM
            else:
                outputs = MODEL(compounds, proteins, morgan_fp, avalon_fp, maccs_fp, protein_features, compounds_bert, proteins_bert)
                predicted_scores = outputs[0]  # 只取final_pred

            loss = LOSS(outputs, labels)  # 使用完整的outputs计算损失
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC



def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_precess(
        MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                       "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        else:
            filepath = save_path + \
                       "/{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test
