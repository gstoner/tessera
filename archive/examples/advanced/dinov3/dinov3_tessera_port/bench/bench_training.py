import os, time, torch
from models.dinov3_tessera.vision_transformer_tsr import VisionTransformerTSR, TileSchedule
from models.dinov3_tessera.dino_head import DINOHead
from models.dinov3_tessera.ssl import DINOSSL, SSLConfig
from models.dinov3_tessera.augment import FakeMultiCropDataset
from torch.utils.data import DataLoader

def run(steps=20, use_kernels=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["TESSERA_USE_CUSTOM_KERNELS"] = "1" if use_kernels else "0"
    os.environ["TESSERA_REFERENCE_KERNELS"] = "0"
    os.environ["TESSERA_NAIVE_ATTENTION"] = "0"

    schedule = TileSchedule()
    s = VisionTransformerTSR(embed_dim=512, depth=8, num_heads=8, schedule=schedule, gram_layers=[4,8]).to(device)
    t = VisionTransformerTSR(embed_dim=512, depth=8, num_heads=8, schedule=schedule, gram_layers=[4,8]).to(device)
    hs = DINOHead(512, out_dim=8192, hidden_dim=1024, bottleneck_dim=256).to(device)
    ht = DINOHead(512, out_dim=8192, hidden_dim=1024, bottleneck_dim=256).to(device)
    ssl = DINOSSL(s, t, hs, ht, SSLConfig(gram_weight=0.0)).to(device)

    opt = torch.optim.AdamW([p for p in ssl.student.parameters()] + list(ssl.head_student.parameters()), lr=1e-3)

    ds = FakeMultiCropDataset(length=100, img_size=224)
    def collate(batch):
        n = len(batch[0][0]); return [torch.stack([b[0][i] for b in batch],0) for i in range(n)]
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate)

    # warmup
    it = iter(dl)
    for _ in range(3):
        views = [v.to(device) for v in next(it)]
        loss, _ = ssl(views)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    start = time.time()
    nstep = 0
    for batch in dl:
        views = [v.to(device) for v in batch]
        loss, _ = ssl(views)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        nstep += 1
        if nstep >= steps: break
    if device=="cuda": torch.cuda.synchronize()
    dt = time.time() - start
    print(f"{'KERNELS' if use_kernels else 'Pytorch'}: {dt/steps*1000:.2f} ms/step over {steps} steps")

if __name__ == "__main__":
    run(steps=20, use_kernels=False)
    run(steps=20, use_kernels=True)
