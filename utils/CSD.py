import torch
import json
import torch.nn.functional as F

class CSD_caculator:
    def __init__(self):
        self.datas = []
        self.labels = []
        self.C = 0
        self.S = 0
        self.D = 0
        self.classCenters = {}
        self.num_class = 0
        self.distr = {}

    def append(self, data, label):
        self.datas.append(data.data.cpu())
        self.labels.append(label.data.cpu())

    def mvdistance(self, x, y):
        return torch.arccos(torch.clamp(torch.mm(x, y.unsqueeze(1)).squeeze(), min=-1,max=1))  # 矩阵乘法替代点积

    def vvdistance(self, x, y):
        return torch.arccos(torch.clamp(torch.dot(x, y), min=-1,max=1))

    def caculate(self):
        datas = torch.cat(self.datas, dim=0)
        datas = F.normalize(datas, dim=1)
        labels = torch.cat(self.labels, dim=0)
        self.num_class = len(labels.unique())
        
        valid_classes = labels.unique().sort().values
        # freq = torch.bincount(labels)
        # freq_dict = {c.item(): freq[c].item() for c in valid_classes}
        # json.dump(freq_dict, open("analysis/freq.json", "w"), indent=4)

        self.distr = {
            c.item(): [] for c in valid_classes
        }
        
        class_centers = torch.stack([
            F.normalize(datas[labels == c].mean(dim=0), dim=0)
            for c in valid_classes
        ])

        dist_matrix = torch.arccos(torch.clamp(datas @ class_centers.T, min=-1, max=1))
        
        # 将标签重新映射到0开始的连续索引
        label_mapping = {c.item(): i for i, c in enumerate(valid_classes)}
        mapped_labels = torch.tensor([label_mapping[c.item()] for c in labels], device=labels.device)
        # 使用映射后的标签
        class_indices = mapped_labels.view(-1, 1)
        diff_matrix = dist_matrix - dist_matrix.gather(1, class_indices)
        
        # 计算类内紧密度 (C)
        intra_class_dists = []
        for i, c in enumerate(valid_classes):
            mask = (labels == c)
            intra_class_dists.append(dist_matrix[mask, i].mean())
            self.distr[c.item()].append(intra_class_dists[i].item())
        self.C = torch.stack(intra_class_dists).mean().item()
        
        # 计算类间分离度 (S)
        inter_class_dists = []
        for i in range(len(valid_classes)):
            tmp = []
            for j in range(len(valid_classes)):
                if j == i:
                    continue
                tmp.append(self.vvdistance(class_centers[i], class_centers[j]))
            self.distr[valid_classes[i].item()].append(torch.stack(tmp).mean().item())
            inter_class_dists.extend(tmp)
        self.S = torch.stack(inter_class_dists).mean().item() if inter_class_dists else 0.0
        
        # 计算分类边界判别性 (D)        
        self.D = torch.stack([(diff_matrix.sum(1) / (len(valid_classes) - 1))[labels == c].mean() for c in valid_classes]).mean().item()

        for c in valid_classes:
            self.distr[c.item()].append(diff_matrix[labels == c].sum(1).mean().item())

    def getResults(self):
        return self.C, self.S, self.D, self.distr
            