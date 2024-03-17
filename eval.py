# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):

        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc

if __name__ == '__main__':
    label_path = '/home/pyw/jieduan2/dataset/test8/lab/' #标签所在文件夹

    # label_path = 'dataset/data5/lab/'
    predict_path = 'submits4/dunet1-8' #测试图片所在文件夹（注意：测试集与标签名称需要一一对应）
    # predict_path = '/home/pyw/mycode/results_prediction/mont_s20_random/contour/'

    pres = os.listdir(predict_path)
    labels = []
    predicts = []
    for im in pres:
        label_name = im.split('.')[0] + '.png'
        # label_name = im.split('.')[0] + '.bmp'
        lab_path = os.path.join(label_path, label_name)
        pre_path = os.path.join(predict_path, im)
        label = cv2.imread(lab_path,0)
        pre = cv2.imread(pre_path,0)
        label[label>0] = 1
        pre[pre>0] = 1
        labels.append(label)
        predicts.append(pre)
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    print('acc: ',acc)
    print('acc_cls: ',acc_cls)
    print('iou: ',iou)
    print('miou: ',miou)
    print('fwavacc: ',fwavacc)

    pres = os.listdir(predict_path)
    init = np.zeros((2,2))
    for im in pres:
        lb_path = os.path.join(label_path, im)
        pre_path = os.path.join(predict_path, im)
        lb = cv2.imread(lb_path, 0)
        pre = cv2.imread(pre_path, 0)
        lb[lb > 0] = 1
        pre[pre > 0] = 1
        pre[0][0] =1
        lb = lb.flatten()
        pre = pre.flatten()
        confuse = confusion_matrix(lb, pre)
        init += confuse

    precision = init[1][1] / (init[0][1] + init[1][1])
    recall = init[1][1] / (init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1]) / init.sum()
    f1_score = 2 * precision * recall / (precision + recall)
    print('precision: ', precision)
    print('class_recall: ', recall)
    print('accuracy: ', accuracy)
    print('f1_score: ', f1_score)
