import torch
from torch.nn.utils import clip_grad_norm_

def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


def train_iter_diffusion(args, in_audio, target_poses, pose_diffusion, optimizer):

    # make pre seq input
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    optimizer.zero_grad()
    pose_diffusion.train()

    loss = pose_diffusion.get_loss(target_poses, pre_seq, in_audio)
    loss.backward()
    clip_grad_norm_(pose_diffusion.parameters(), 10)
    optimizer.step()

    ret_dict = {'loss':loss.item()}
    return ret_dict

