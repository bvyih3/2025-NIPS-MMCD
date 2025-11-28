import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
import numpy as np
from torch.autograd import grad
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)
# from tensorboardX import SummaryWriter
# writer_tsne = SummaryWriter('runs/tsne')
from utils.CSD import CSD_caculator
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def compute_supcon_loss(feats, targets, tau, weights=None):
    feats_filt = F.normalize(feats, dim=1)
    targets_r = targets.reshape(-1, 1)
    targets_c = targets.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    if weights is not None:
        weights = weights.repeat(feats_sim.shape[0], 1)
        feats_sim = feats_sim * weights
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives, 1)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim = 1)
    pred = pred.detach().cpu().numpy()
    score = (pred == np.array(labels))
    tot_correct = score.sum()
    return tot_correct


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores
    
def compute_loss(output, labels):

    #Function for calculating loss
    
    ce_loss = nn.CrossEntropyLoss(reduction='mean')(output, labels.squeeze(-1).long())
    
    return ce_loss


def train(model, m_model, optim, train_loader, loss_fn, tracker, writer, tb_count, epoch, args):

    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))

    for v, q, a, mg, q_id, f1, type, a_type in loader:
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        mg = mg.cuda()
        f1 = f1.cuda()
        gt = torch.argmax(a, 1)
        hidden_, ce_logits, q_logits, v_logits, q_repr, v_repr = model(v, q)

        out_q = (torch.mm(torch.mm(q_repr, model.qweight.main[0].weight.t()) + model.qweight.main[0].bias / 2, model.qweight.main[3].weight.t())
                    + model.qweight.main[3].bias / 2)
        out_v = (torch.mm(torch.mm(v_repr, model.vweight.main[0].weight.t()) + model.vweight.main[0].bias / 2, model.vweight.main[3].weight.t())
                    + model.vweight.main[3].bias / 2)
        score_q = torch.softmax(out_q, 1)
        score_v = torch.softmax(out_v, 1)

        mask = (ce_logits+score_q+score_v) != 0
        mod_logits = (ce_logits * ce_logits + score_q * q_logits + score_v * v_logits)
        mod_logits[mask] = mod_logits[mask] / (ce_logits + score_q + score_v)[mask]
        mod_logits[~mask] = 0

        hidden, pred, batch_d = m_model(hidden_, ce_logits, mod_logits, mg, epoch, a, args)
        
        dict_args = {'margin': mg, 'hidden': hidden, 'warm': epoch >= args.w, 'per': f1, 'diff': batch_d}
                
        loss = loss_fn(hidden, a, **dict_args)
        ce_loss = - F.log_softmax(ce_logits, dim=-1) * a
        ce_loss = ce_loss * f1
        loss += ce_loss.sum(dim=-1).mean()

        if epoch < args.w:
            supcon_loss = args.lambda3*compute_supcon_loss(hidden_, gt, tau = args.tau1)
        else:
            weights = torch.exp(batch_d)
            supcon_loss = args.lambda3*compute_supcon_loss(hidden_, gt, args.tau1, weights)
        loss = loss + supcon_loss

        # writer.add_scalars('data/losses', {
        # }, tb_count)
        tb_count += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()
        
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                           acc=fmt(acc_trk.mean.value))

    return tb_count

#Evaluation code
def evaluate(model, m_model, dataloader, args, epoch=0, write=False):
    score = 0
    qat_score = {}
    qat_total = {}

    for v, q, a, mg, q_id, f1, qtype, a_type in tqdm(dataloader, ncols=0, leave=True):
        v = v.cuda()
        q = q.cuda()
        mg = mg.cuda()
        a = a.cuda()
        gt = torch.argmax(a, 1)
        hidden_, ce_logits, q_logits, v_logits, q_repr, v_repr = model(v, q)
        hidden, pred, batch_diff_res = m_model(hidden_, ce_logits, ce_logits, mg, epoch, a, None)

        # if write:
        #     results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum(1)
        score += batch_score.sum()

        # for atype, idx in atype2idx.items():
        #     mask = a_type == idx
        #     qat_score[atype] = qat_score.get(atype, 0) + batch_score[mask].sum()
        #     qat_total[atype] = qat_total.get(atype, 0) + mask.sum()
        
        
    print(score, len(dataloader.dataset))
    score = score / len(dataloader.dataset)

    # for key in qat_score:
    #     print(key + ": " + str((qat_score[key]/qat_total[key]*100).item()))

    # if write:
    #     print("saving prediction results to disk...")
    #     result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
    #         config.task, config.test_split, config.version, epoch)
    #     with open(result_file, 'w') as fd:
    #         json.dump(results, fd)
    print(score)

    return score
