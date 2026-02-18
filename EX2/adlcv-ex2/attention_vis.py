import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from imageclassification import prepare_dataloaders, set_seed, select_two_classes_from_cifar10

def attention_rollout(attn_list):
    result = None

    for attn in attn_list:
        # prosjek preko head-ova
        attn = attn.mean(dim=1)  # (B, N, N)

        # dodaj identity (residual connection)
        N = attn.size(-1)
        identity = torch.eye(N).to(attn.device)
        attn = attn + identity

        # normalizuj redove
        attn = attn / attn.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn
        else:
            result = attn @ result

    return result

def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=256, num_heads=8, num_layers=6, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         batch_size=16):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        channels=channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pos_enc=pos_enc,
        pool=pool,
        dropout=dropout,
        fc_dim=fc_dim,
        num_classes=num_classes
    ).to(device)

    # 🔥 Učitaj istreniran model
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    images, labels = next(iter(test_iter))
    images = images[:5].to(device)

    with torch.no_grad():
        _ = model(images)

    attn_list = [block.last_attn for block in model.transformer_blocks]
    rollout = attention_rollout(attn_list)

    H, W = image_size
    patch_h, patch_w = patch_size
    nph = H // patch_h
    npw = W // patch_w

    fig, axes = plt.subplots(1, 5, figsize=(20,4))

    for i in range(5):

        cls_attn = rollout[i, 0, 1:]
        attn_map = cls_attn.reshape(nph, npw)

        attn_map = attn_map.unsqueeze(0).unsqueeze(0)
        attn_map = F.interpolate(attn_map, size=(H, W), mode='bilinear')
        attn_map = attn_map.squeeze().cpu()

        # Normalizuj attention map između 0 i 1
        attn_map = attn_map - attn_map.min()
        attn_map = attn_map / (attn_map.max() + 1e-8)

        img_vis = images[i].permute(1,2,0).cpu()

        # proširi attention na 3 kanala
        attn_map_3c = attn_map.unsqueeze(-1).repeat(1,1,3)

        masked_img = img_vis * attn_map_3c

        axes[i].imshow(masked_img)
        axes[i].set_title(f"Image {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("attention_rollout_masked.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()