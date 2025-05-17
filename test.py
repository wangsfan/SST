import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from SSTC.dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import torch
from torch.nn import functional as F



def detailed_test(args, step, test_data_set):
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
    causal = torch.load('./save/{}/causal_{}.pth'.format(args.dataset, step))
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    all_test_out_logits = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            test_visual = test_data[0]
            test_audio = test_data[1]
            if causal[step] is None:
                test_causal = None
            else:
                test_causal_feature = causal[step].unsqueeze(0).repeat_interleave(test_audio.shape[0], dim=0)
                audio_features = test_audio.unsqueeze(-2)  # 变为 batch×1×768
                test_causal = torch.cat((test_causal_feature, audio_features), dim=1).to(device)  # 变为 batch×8×768
            test_visual = test_visual.to(device)
            test_audio = test_audio.to(device)
            test_out_logits = model(visual=test_visual, audio=test_audio, causal=test_causal)
            test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
            all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
    print("Incremental step {} Testing res: {:.6f}".format(step, test_top1))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ksounds', choices=['AVE', 'ksounds', 'VGGSound_100'])
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['audio-visual'])

    parser.add_argument('--infer_batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=0)



    args = parser.parse_args()
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step
    test_set = IcaAVELoader(args=args, mode='test', modality=args.modality)

    for step in range(total_incremental_steps):
        test_set.set_incremental_step(step)
        detailed_test(args, step, test_set)
    
    if args.dataset != 'AVE':
        test_set.close_visual_features_h5()
    

