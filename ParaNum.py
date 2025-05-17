from torchsummary import summary
from SSTC.audio_visual_model_incremental import IncreAudioVisualNet
import torch
from thop import profile
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ksounds', choices=['AVE', 'ksounds', 'VGGSound_100'])
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['audio-visual'])
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--infer_batch_size', type=int, default=128)
    parser.add_argument('--exemplar_batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, default=[100], nargs='+', help="")

    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--lam_a', type=float, default=0.5)
    parser.add_argument('--lam_I', type=float, default=0.1)
    parser.add_argument('--lam_C', type=float, default=1.0)
    parser.add_argument('--mask_probability', type=float, default=0.5)
    parser.add_argument('--num_idx', type=int, default=23)
    parser.add_argument('--seed', type=int, default=115411)

    parser.add_argument('--class_num_per_step', type=int, default=6)

    parser.add_argument('--memory_size', type=int, default=500)

    parser.add_argument('--instance_contrastive', action='store_true', default=True)
    parser.add_argument('--class_contrastive', action='store_true', default=True)
    parser.add_argument('--attn_score_distil', action='store_true', default=False)
    parser.add_argument('--audio_KD', action='store_true', default=True)

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.05)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.05)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IncreAudioVisualNet(args, 7)
    visual = torch.zeros((1,1568, 768)).to(device)
    audio = torch.zeros((1,768)).to(device)
    model = model.to(device)
    flops , param = profile(model,inputs=(visual,audio))
    print(flops/1000000000 , param/10000000)