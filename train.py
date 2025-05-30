import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from SSTC.dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from SSTC.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
from itertools import cycle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr

def train(args, step, train_data_set, val_data_set, exemplar_set,causal):
    T = 2

    train_loader = DataLoader(train_data_set, batch_size=min(args.train_batch_size, train_data_set.__len__()), num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=min(args.infer_batch_size, val_data_set.__len__()), num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
    else:
        model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step-1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step-1))

        exemplar_loader = DataLoader(exemplar_set, batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()), num_workers=args.num_workers,
                                     pin_memory=True, drop_last=True, shuffle=True)

        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)
    
    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        # old_model = old_model.to('cpu')
        old_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()
        if step == 0:
            iterator = tqdm(train_loader)
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader))
        
        for samples in iterator:
            if step == 0:
                data, labels = samples
                labels = labels.to(device)
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                out, audio_feature, visual_feature = model(visual=visual, audio=audio, out_feature_before_fusion=True)
                # CE_loss = CE_loss(step_out_class_num, out, labels)
                # loss = CE_loss
                loss = CE_loss(step_out_class_num, out, labels)
            else:
                curr, prev = samples
                data, labels = curr
                # labels = labels % ((step_out_class_num - 1) - (last_step_out_class_num - 1))
                labels = labels.to(device)
                labels_ = labels % args.class_num_per_step
                labels_ = labels_.to(device)

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device)

                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                audio = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_audio = exemplar_data[1]
                total_visual = torch.cat((visual, exemplar_visual))
                total_audio = torch.cat((audio, exemplar_audio))
                causal_feature = causal[step].unsqueeze(0).repeat_interleave(total_audio.shape[0], dim=0)
                audio_features = total_audio.unsqueeze(-2)  # 变为 batch×1×768

                if step > 1:
                    old_causal_feature = causal[step - 1].unsqueeze(0).repeat_interleave(total_audio.shape[0], dim=0)
                    old_audio_features = total_audio.unsqueeze(-2)  # 变为 batch×1×768
                    old_total_causal = torch.cat((old_causal_feature, old_audio_features), dim=1).to(
                        device)  # 变为 batch×8×768

                total_causal = torch.cat((causal_feature, audio_features), dim=1).to(device)
                total_visual = total_visual.to(device)
                total_audio = total_audio.to(device)
                out, audio_feature, visual_feature, cau_audio,ori_audio,spatial_attn_score, temporal_attn_score = model(visual=total_visual, audio=total_audio,
                                                                                                    causal=total_causal,
                                                                                                    out_feature_before_fusion=True, out_attn_score=True,
                                                                                                    out_audio_norm=True)
                with torch.no_grad():
                    if step == 1:
                        old_total_causal = None
                    old_out, old_audio_feature, old_visual_feature, old_cau_audio, old_ori_audio, old_spatial_attn_score, old_temporal_attn_score = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        causal=old_total_causal,
                        out_feature_before_fusion=True,
                        out_attn_score=True,
                        out_audio_norm=True)
                    old_out = old_out.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()
                    old_cau_audio = old_cau_audio.detach()
                
                if args.instance_contrastive:
                    instance_contra_loss = cal_contrastive_loss(audio_feature, visual_feature, temperature=args.instance_contrastive_temperature)
                
                if args.class_contrastive:
                    all_labels = torch.cat((labels, exemplar_labels))
                    class_contra_loss = class_contrastive_loss(audio_feature, visual_feature, all_labels, temperature=args.class_contrastive_temperature)
                
                if args.attn_score_distil:
                    exem_spatial_attn_score = spatial_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(2, 3)
                    exem_spatial_attn_score = exem_spatial_attn_score.reshape(-1, exem_spatial_attn_score.shape[-1])

                    exem_old_spatial_attn_score = old_spatial_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(2, 3)
                    exem_old_spatial_attn_score = exem_old_spatial_attn_score.reshape(-1, exem_old_spatial_attn_score.shape[-1])

                    exem_temporal_attn_score = temporal_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(1, 2)
                    exem_temporal_attn_score = exem_temporal_attn_score.reshape(-1, exem_temporal_attn_score.shape[-1])

                    exem_old_temporal_attn_score = old_temporal_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(1, 2)
                    exem_old_temporal_attn_score = exem_old_temporal_attn_score.reshape(-1, exem_old_temporal_attn_score.shape[-1])

                    spatial_attn_dist_loss = F.kl_div(exem_spatial_attn_score.log(), exem_old_spatial_attn_score, reduction='sum') / exemplar_data_batch_size
                    temporal_attn_dist_loss = F.kl_div(exem_temporal_attn_score.log(), exem_old_temporal_attn_score, reduction='sum') / exemplar_data_batch_size

                if args.audio_KD:
                    # 旧任务蒸馏
                    exm_old_audio_feature=old_cau_audio[
                                              data_batch_size:data_batch_size + exemplar_data_batch_size]
                    exm_audio_feature = cau_audio[
                                            data_batch_size:data_batch_size + exemplar_data_batch_size]
                    # 新任务
                    c_ori_audio = cau_audio[:data_batch_size]
                    ori_audio = ori_audio[:data_batch_size]
                    audio_dist_loss1 = corr_loss(exm_old_audio_feature, exm_audio_feature)
                    audio_dist_loss2 = corr_loss(ori_audio, c_ori_audio)
                    audio_dist_loss = audio_dist_loss1+audio_dist_loss2

                old_out = old_out[:,:last_step_out_class_num]
                
                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(args.class_num_per_step, curr_out, labels_)

                prev_out = out[data_batch_size:data_batch_size+exemplar_data_batch_size, :last_step_out_class_num]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)

                loss_CE = (loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size) / (data_batch_size + exemplar_data_batch_size)

                if args.dataset == 'AVE' and args.class_num_per_step == 4 and step == 1:
                    loss_CE = CE_loss(args.class_num_per_step + last_step_out_class_num, out, torch.cat((labels, exemplar_labels)))

                loss_KD = torch.zeros(step).to(device)
                
                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step

                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                loss_KD = loss_KD.sum()
                loss = loss_CE + loss_KD
                if args.instance_contrastive:
                    loss += args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss += args.lam_C * class_contra_loss
                if args.attn_score_distil:
                    loss += args.lam * spatial_attn_dist_loss + (1 - args.lam) * temporal_attn_dist_loss
                if args.audio_KD:
                    loss += args.lam_a * audio_dist_loss
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0]
                val_audio = val_data[1]
                if causal[step] is None:
                    val_causal = None
                else:
                    val_causal_feature = causal[step].unsqueeze(0).repeat_interleave(val_audio.shape[0], dim=0)
                    audio_features = val_audio.unsqueeze(-2)  # 变为 batch×1×768
                    val_causal = torch.cat((val_causal_feature, audio_features), dim=1).to(device)  # 变为 batch×8×768
                val_visual = val_visual.to(device)
                val_audio = val_audio.to(device)
                if torch.cuda.device_count() > 1:
                    val_out_logits = model.module.forward(visual=val_visual, audio=val_audio, causal=val_causal)
                else:
                    val_out_logits = model(visual=val_visual, audio=val_audio, causal=val_causal)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1: 
                torch.save(model.module, './save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
                torch.save(causal, './save/{}/causal_{}.pth'.format(args.dataset, step))
            else:
                torch.save(model, './save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
                torch.save(causal, './save/{}/causal_{}.pth'.format(args.dataset, step))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/train_loss_step_{}.png'.format(args.dataset, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/val_acc_step_{}.png'.format(args.dataset, step))
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ksounds', choices=['AVE', 'ksounds', 'VGGSound_100'])
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['audio-visual'])
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--infer_batch_size', type=int, default=128)
    parser.add_argument('--exemplar_batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[100], nargs='+', help="")
    
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--lam_a', type=float, default=0.6)
    parser.add_argument('--lam_I', type=float, default=0.1)
    parser.add_argument('--lam_C', type=float, default=1.0)
    parser.add_argument('--mask_probability', type=float, default=0.6)
    parser.add_argument('--num_idx', type=int, default=45)
    parser.add_argument('--seed', type=int, default=115411)

    parser.add_argument('--class_num_per_step', type=int, default=6)

    parser.add_argument('--memory_size', type=int, default=500)

    parser.add_argument('--instance_contrastive', action='store_true', default=True)
    parser.add_argument('--class_contrastive', action='store_true', default=True)
    parser.add_argument('--attn_score_distil', action='store_true', default=True)
    parser.add_argument('--audio_KD', action='store_true', default=True)

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.05)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.05)
    

    args = parser.parse_args()
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)
    
    print('Training start time: {}'.format(datetime.now()))

    train_set = IcaAVELoader(args=args, mode='train', modality=args.modality)
    val_set = IcaAVELoader(args=args, mode='val', modality=args.modality)

    exemplar_set = exemplarLoader(args=args, modality=args.modality)


    task_best_acc_list = []

    step_forgetting_list = []
    causal = []
    causal.append(None)
    
    exemplar_class_vids = None
    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)

        exemplar_set._set_incremental_step_(step)
        if step>0:
            causal.append(exemplar_set.set_causal(step))

        print('Incremental step: {}'.format(step))

        train(args, step, train_set, val_set, exemplar_set,causal)
    
    if args.dataset != 'AVE':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        exemplar_set.close_visual_features_h5()
    

