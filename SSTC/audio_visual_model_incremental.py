import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LSCLinear, SplitLSCLinear


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNet, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num
        self.num_idx = args.num_idx
        self.mask_probability = args.mask_probability
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')
        if self.modality == 'visual':
            self.visual_proj = nn.Linear(768, 768)
        elif self.modality == 'audio':
            self.audio_proj = nn.Linear(768, 768)
        else:
            self.audio_proj = nn.Linear(768, 768)
            self.visual_proj = nn.Linear(768, 768)
            self.attn_audio_proj = nn.Linear(768, 768)
            self.attn_visual_proj = nn.Linear(768, 768)
            self.feature_proj = nn.Linear(768, 768)
            self.feature_proj2 = nn.Linear(768, 1)
        
        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)
        self.causal_param = nn.Parameter(data=torch.zeros(1), requires_grad=True)
    
    def forward(self, visual=None, audio=None,causal=None ,out_logits=True, out_features=False, out_features_norm=False, out_audio_norm=False,out_feature_before_fusion=False, out_attn_score=False, AFC_train_out=False,mode='Train'):
        if visual is None:
            raise ValueError('input frames are None when modality contains visual')
        if audio is None:
            raise ValueError('input audio are None when modality contains audio')

        visual = visual.view(visual.shape[0], 8, -1, 768)
        ori_audio = audio
        if causal is None:
            spatial_filtered_score, temporal_filtered_score= self.audio_visual_attention2(audio, visual)
            spatial_attn_score = spatial_filtered_score
            temporal_attn_score = temporal_filtered_score
        else:
            # audio = self.causal_intervention_mean_feature(causal)
            audio = self.causal_intervention_mean_feature4(causal,ori_audio)
            spatial_filtered_score, temporal_filtered_score,spatial_attn_score,temporal_attn_score = self.audio_visual_attention1(audio, visual)
        visual_pooled_feature = torch.sum(spatial_filtered_score * visual, dim=2)
        visual_pooled_feature = torch.sum(temporal_filtered_score * visual_pooled_feature, dim=1)

        audio_feature = F.relu(self.audio_proj(audio))
        visual_feature = F.relu(self.visual_proj(visual_pooled_feature))
        audio_visual_features = visual_feature + audio_feature

        logits = self.classifier(audio_visual_features)
        outputs = ()
        if AFC_train_out:
            audio_feature.retain_grad()
            visual_feature.retain_grad()
            visual_pooled_feature.retain_grad()
            outputs += (logits, visual_pooled_feature, audio_feature, visual_feature)
            return outputs
        else:
            if out_logits:
                outputs += (logits,)
            if out_features:
                if out_features_norm:
                    outputs += (F.normalize(audio_visual_features),)
                else:
                    outputs += (audio_visual_features,)
            if out_feature_before_fusion:
                outputs += (F.normalize(audio_feature), F.normalize(visual_feature))
            if out_audio_norm:
                outputs += (F.normalize(audio),F.normalize(ori_audio))
            if out_attn_score:
                outputs += (spatial_attn_score, temporal_attn_score)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs

    def audio_visual_attention1(self, audio_features, visual_features):

        # 设置阈值
        mask_probability = self.mask_probability  # 掩码概率
        num_idx = self.num_idx

        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        # (BS, 8, 14*14, 768) 768个维度中，每个维度中196个空间patch的得分总和为1
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # spa_mean = torch.mean(torch.sum(spatial_attn_score,dim=-1),dim=-1).unsqueeze(-1)
        spatial_attn_patch_score = torch.sum(spatial_attn_score,dim=-1)  # /spa_mean # BS,8,14*14 torch.Size([128, 8, 196])
        sorted_tensor, indices = torch.sort(spatial_attn_patch_score, dim=-1)
        sorted_tensor = sorted_tensor[:, :, num_idx]
        # print("1:",sorted_tensor.size()) #1: torch.Size([128, 8])
        sorted_tensor=sorted_tensor.unsqueeze(-1).repeat(1, 1, 196)
        # print("2:", sorted_tensor.size(),spatial_attn_patch_score.size())
        # # print("1:",spa_mean)
        # # 设置掩码逻辑：小于阈值的元素设置为 0，其他保留
        spa_threshold_mask = (spatial_attn_patch_score < sorted_tensor).unsqueeze(-1).repeat(1, 1, 1, 768) # d大于阈值为前景flase 小于阈值为背景true
        # # print("2:",spa_threshold_mask.size(), spa_threshold_mask[0][0][:5])
        random_mask = torch.rand_like(spatial_attn_score) < mask_probability # 小于mask_probability为true
        final_mask = ~(spa_threshold_mask & random_mask).to(spatial_attn_score.device) #  (同时满足阈值条件和随机条件的位置为 True,即背景以一定的概率被mask。再取反)
        spatial_filtered_score = final_mask*spatial_attn_score
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_filtered_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1) #
        return spatial_filtered_score, temporal_attn_score,spatial_attn_score,temporal_attn_score

    def audio_visual_attention2(self, audio_features, visual_features):
        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        # (BS, 8, 14*14, 768)
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score

    #
    def causal_intervention_mean_feature4(self,causal_feature,ori_audio):
        # causal_param = torch.sigmoid(self.causal_param)
        causal_param = self.soft_clamp2(self.causal_param)
        features = self.feature_proj(causal_feature)
        features2 = self.feature_proj2(causal_feature).view(causal_feature.shape[0], 1, -1)# 1. batch×10×1 2.batch×1×10
        feature_weight = F.softmax(torch.tanh(features),dim=1) # 1.变为 batch×10×768 2. 将8个系数softmax
        causal_inter_feature = feature_weight * causal_feature #变为 batch×10×768
        causal_inter_feature = torch.bmm(features2, causal_inter_feature).squeeze(-2)
        # feature_weight * causal_feature batch×10×768
        return causal_param*causal_inter_feature+ori_audio

    def soft_clamp2(self,x):
        return torch.sigmoid(x )

    def incremental_classifier(self, numclass):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, numclass, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias

