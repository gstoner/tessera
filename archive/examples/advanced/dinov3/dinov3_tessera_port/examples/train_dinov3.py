import argparse, os, time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models.dinov3_tessera.vision_transformer_tsr import VisionTransformerTSR, TileSchedule
from models.dinov3_tessera.dino_head import DINOHead
from models.dinov3_tessera.ssl import DINOSSL, SSLConfig
from models.dinov3_tessera.augment import MultiCropTransform, FakeMultiCropDataset


def collate_views(batch):
    n_views = len(batch[0][0])
    out = []
    for i in range(n_views):
        out.append(torch.stack([sample[0][i] for sample in batch], dim=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--data", type=str, default=None, help="ImageNet-style root (train folder)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=0, help="override number of steps for quick tests")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gram-weight", type=float, default=0.0)
    ap.add_argument("--gram-layers", type=str, default="", help="comma-separated layer indices, e.g. 4,8")
    args = ap.parse_args()

    transform = MultiCropTransform(global_size=224, local_size=96, n_local=6)

    if args.fake_data or args.data is None:
        ds = FakeMultiCropDataset(length=64, img_size=224, transform=transform)
    else:
        train_dir = os.path.join(args.data, "train") if os.path.isdir(os.path.join(args.data, "train")) else args.data
        base = torchvision.datasets.ImageFolder(train_dir, transform=None)
        class Wrap(torch.utils.data.Dataset):
            def __init__(self, base, tf): self.base = base; self.tf = tf
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                img, _ = self.base[i]
                return self.tf(img), 0
        ds = Wrap(base, transform)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_views)

    schedule = TileSchedule(block_m=128, block_n=128, block_k=64, stages=2, num_warps=8, smem_bytes=128*1024)
    gram_layers = [int(x) for x in args.gram_layers.split(",") if x.strip().isdigit()] if args.gram_layers else []

    student = VisionTransformerTSR(img_size=224, patch_size=16, embed_dim=512, depth=8, num_heads=8, schedule=schedule, gram_layers=gram_layers)
    teacher = VisionTransformerTSR(img_size=224, patch_size=16, embed_dim=512, depth=8, num_heads=8, schedule=schedule, gram_layers=gram_layers)
    head_s = DINOHead(512, out_dim=8192, hidden_dim=1024, bottleneck_dim=256, nlayers=3)
    head_t = DINOHead(512, out_dim=8192, hidden_dim=1024, bottleneck_dim=256, nlayers=3)

    cfg = SSLConfig(student_temp=0.1, teacher_temp=0.04, ema_momentum=0.996, gram_weight=args.gram_weight, gram_layers=gram_layers)
    model = DINOSSL(student, teacher, head_s, head_t, cfg).to(args.device)

    opt = torch.optim.AdamW([p for p in model.student.parameters()] + list(model.head_student.parameters()), lr=args.lr, weight_decay=0.05)

    global_step = 0
    max_steps = args.steps if args.steps > 0 else len(dl) * args.epochs
    for epoch in range(args.epochs):
        for batch in dl:
            views = [v.to(args.device) for v in batch]
            loss, metrics = model(views)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % 5 == 0:
                print(f"step {global_step}/{max_steps}  loss={loss.item():.4f}  ce={metrics['loss_ce'].item():.4f}  gram={metrics['loss_gram'].item():.4f}")
            if args.steps and global_step >= args.steps:
                break
        if args.steps and global_step >= args.steps:
            break

    print("Done.")

if __name__ == "__main__":
    main()
