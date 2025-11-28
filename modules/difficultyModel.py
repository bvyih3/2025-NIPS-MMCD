import torch.nn as nn
import torch
import torch.nn.functional as F


class DifficultyModel(nn.Module):
    def __init__(self, datasetSize, num_ans_candidates, args):
        super(DifficultyModel, self).__init__()
        self.epsilon = args.c
        # self.d_nu = torch.zeros(datasetSize).cuda()
        # self.d_de = torch.zeros(datasetSize).cuda()
        # self.d_margin_nu = torch.zeros(datasetSize).cuda()
        # self.d_margin_de = torch.zeros(datasetSize).cuda()
        self.d_nu = torch.ones(datasetSize).cuda() * self.epsilon
        self.d_de = torch.ones(datasetSize).cuda() * self.epsilon
        self.d_margin_nu = torch.ones(datasetSize).cuda() * self.epsilon
        self.d_margin_de = torch.ones(datasetSize).cuda() * self.epsilon
        self.lastpred = torch.ones(datasetSize, num_ans_candidates) / num_ans_candidates
        self.lastpred = self.lastpred.cuda()
        self.lastpred.requires_grad = False
        self.d_nu.requires_grad = False
        self.d_de.requires_grad = False
        self.d_margin_nu.requires_grad = False
        self.d_margin_de.requires_grad = False
        self.alpha = args.alpha
        self.beta = args.beta
        self.add = args.add
        self.entryIdx = 0
        self.datasetSize = datasetSize

    def forward(self, pred, a):
        batch_size = pred.shape[0]
        lastp = self.lastpred[self.entryIdx : self.entryIdx + batch_size]

        gt = torch.argmax(a, 1)        
        p = F.softmax(pred, dim=1)
        m = (p + lastp) / 2
        p_gt = p.gather(1, gt.unsqueeze(1)).squeeze(1)
        lastp_gt = lastp.gather(1, gt.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 向量化计算JSD差异
            d = 0.5 * lastp * torch.log(lastp / m) + 0.5 * p * torch.log(p / m)
            # 创建类别掩码
            class_mask = torch.arange(a.shape[1], device=lastp.device).expand(batch_size, -1)
            gt_mask = class_mask == gt.unsqueeze(1)  # 真实类别掩码
            
            # 条件1: 真实类别且lastp_gt > p_gt
            cond1 = gt_mask & (lastp_gt.unsqueeze(1) > p_gt.unsqueeze(1))
            
            # 条件2: 非真实类别且lastp < p
            cond2 = ~gt_mask & (lastp < p)
            
            # 合并条件并向量化求和
            du = (d * (cond1 | cond2).float()).sum(dim=1)
            d = torch.sum(d, 1)
            dl = d - du

            # 向量化计算margin
            sorted_p, _ = torch.sort(p, dim=1, descending=True)
            gt_expanded = p_gt.view(-1, 1).expand_as(sorted_p)
            
            # 计算三个部分的mask
            greater_mask = (sorted_p > gt_expanded)
            less_mask = (sorted_p < gt_expanded)
            
            # 向量化计算各部分的值
            greater_values = (sorted_p - gt_expanded) * greater_mask
            less_values = (gt_expanded - sorted_p) * less_mask

            # 总和计算
            unlearn = torch.exp(greater_values).sum(1) - torch.sum(~greater_mask, 1)
            nozero_greater_mask = torch.sum(greater_mask, 1) > 0
            unlearn[nozero_greater_mask] = torch.log(unlearn[nozero_greater_mask] / torch.sum(greater_mask, 1)[nozero_greater_mask])

            nozero_less_mask = torch.sum(less_mask, 1) > 0
            learn = torch.exp(less_values).sum(1) - torch.sum(~less_mask, 1)
            learn[nozero_less_mask] = torch.log(learn[nozero_less_mask] / torch.sum(less_mask, 1)[nozero_less_mask])

            if self.add:
                self.d_nu[self.entryIdx : self.entryIdx + batch_size] += du
                self.d_de[self.entryIdx : self.entryIdx + batch_size] += dl
                self.d_margin_nu[self.entryIdx : self.entryIdx + batch_size] += unlearn
                self.d_margin_de[self.entryIdx : self.entryIdx + batch_size] += learn
            else:
                self.d_nu[self.entryIdx : self.entryIdx + batch_size] = (1 - self.beta) * self.d_nu[self.entryIdx : self.entryIdx + batch_size] + self.beta * du
                self.d_de[self.entryIdx : self.entryIdx + batch_size] = (1 - self.beta) * self.d_de[self.entryIdx : self.entryIdx + batch_size] + self.beta * dl
                self.d_margin_nu[self.entryIdx : self.entryIdx + batch_size] = (1 - self.beta) * self.d_margin_nu[self.entryIdx : self.entryIdx + batch_size] + self.beta * unlearn
                self.d_margin_de[self.entryIdx : self.entryIdx + batch_size] = (1 - self.beta) * self.d_margin_de[self.entryIdx : self.entryIdx + batch_size] + self.beta * learn

        part1 = self.d_nu[self.entryIdx : self.entryIdx + batch_size] / self.d_de[self.entryIdx : self.entryIdx + batch_size]
        part2 = self.d_margin_nu[self.entryIdx : self.entryIdx + batch_size] / self.d_margin_de[self.entryIdx : self.entryIdx + batch_size]
        batch_d = (1 - self.alpha) * part1 + self.alpha * part2
        self.lastpred[self.entryIdx : self.entryIdx + batch_size] = p.clone().detach()
        self.entryIdx = (self.entryIdx + batch_size) % self.datasetSize
        return batch_d