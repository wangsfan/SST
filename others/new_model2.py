import torch
from torch import nn
import torch.nn.functional as F
from .models2 import New_Audio_Guided_Attention
from .layers import LSCLinear, SplitLSCLinear


class IncreAudioVisualNet(nn.Module):
    def __init__( self,args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNet, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num
        self.spatial_channel_att = New_Audio_Guided_Attention().cuda()  # batch * 10,512
        self.video_input_dim = 768
        self.video_fc_dim = 768
        self.d_model = 768
        self.relu = nn.ReLU()
        self.causal_param = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.feature_proj = nn.Linear(768, 768)
        self.feature_proj2 = nn.Linear(768, 1)



        self.audio_proj = nn.Linear(768, 768)
        self.visual_proj = nn.Linear(768, 768)

        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)

    def forward(self, visual=None, audio=None, causal=None, out_logits=True, out_features=False,
                out_features_norm=False, out_audio_norm=False, out_feature_before_fusion=False, out_attn_score=False,
                AFC_train_out=False, mode='Train'):
        if visual is None:
            raise ValueError('input frames are None when modality contains visual')
        if audio is None:
            raise ValueError('input audio are None when modality contains audio')
        # print(visual.size(),audio.size())

        visual = visual.view(visual.shape[0], 8, -1, 768)
        ori_audio = audio
        if causal is None:
            visual_feature,spatial_att_maps,temporal_att_maps = self.spatial_channel_att(visual)
        else:
            audio = self.causal_intervention_mean_feature4(causal, ori_audio)
            visual_feature,spatial_att_maps,temporal_att_maps = self.spatial_channel_att(visual)

        audio_feature = F.relu(self.audio_proj(audio))
        visual_feature = F.relu(self.visual_proj(visual_feature))
        audio_visual_features = visual_feature + audio_feature

        logits = self.classifier(audio_visual_features)
        outputs = ()
        if AFC_train_out:
            audio_feature.retain_grad()
            visual_feature.retain_grad()
            outputs += (logits, audio_feature, visual_feature)
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
                outputs += (F.normalize(audio), F.normalize(ori_audio))
            if out_attn_score:
                outputs += (spatial_att_maps, temporal_att_maps)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs

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
