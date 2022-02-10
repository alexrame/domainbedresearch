import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

class EMA:
    def __init__(self, alpha=0.9, num_labels=1):
        # self.label = label
        self.alpha = alpha
        self.parameter = np.zeros(num_labels)
        # self.parameter = torch.zeros(label.size(0))
        # self.updated = torch.zeros(label.size(0))
        self.updated = np.zeros(num_labels)
        self.label = np.zeros(num_labels)

    def update(self, data, index, label):
        max_index = index.max().item()
        # index = index.cpu().numpy()
        data = data.cpu().numpy()
        # print(max_index)
        if max_index >= len(self.label):
            self.label = np.pad(self.label, (0, 1+ max_index - len(self.label)))
            self.parameter = np.pad(self.parameter, (0, 1+ max_index - len(self.parameter)))
            self.updated = np.pad(self.updated, (0, 1+ max_index - len(self.updated)))

        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1
        self.label[index] = label.cpu().numpy()

    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()

# class EMA:
#     def __init__(self, label, alpha=0.9):
#         self.label = label
#         self.alpha = alpha
#         self.parameter = torch.zeros(label.size(0))
#         self.updated = torch.zeros(label.size(0))
        
#     def update(self, data, index):
#         self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
#         self.updated[index] = 1
        
#     def max_loss(self, label):
#         label_index = np.where(self.label == label)[0]
#         return self.parameter[label_index].max()
