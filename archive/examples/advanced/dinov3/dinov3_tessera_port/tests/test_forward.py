import torch
from models.dinov3_tessera.vision_transformer_tsr import VisionTransformerTSR
from models.dinov3_tessera.dino_head import DINOHead
from models.dinov3_tessera.ssl import DINOSSL, SSLConfig
from models.dinov3_tessera.ops.flash_attention_tessera import TileLinear, tessera_layer_norm, tessera_flash_attn, DummySchedule

def test_ssl_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 224, 224, device=device)
    s = VisionTransformerTSR(embed_dim=256, depth=4, num_heads=4, gram_layers=[2,4]).to(device)
    t = VisionTransformerTSR(embed_dim=256, depth=4, num_heads=4, gram_layers=[2,4]).to(device)
    hs = DINOHead(256, out_dim=4096, hidden_dim=512, bottleneck_dim=128).to(device)
    ht = DINOHead(256, out_dim=4096, hidden_dim=512, bottleneck_dim=128).to(device)
    ssl = DINOSSL(s, t, hs, ht, SSLConfig(gram_weight=0.05, gram_layers=[2,4])).to(device)
    loss, metrics = ssl([x, x, x, x])
    assert torch.isfinite(loss), "loss should be finite"
    print("ssl ok", {k: float(v) for k,v in metrics.items()})

def test_tilelinear_matches_linear():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B,N,K,M = 2, 5, 17, 13
    x = torch.randn(B,N,K, device=device)
    lin = torch.nn.Linear(K, M, bias=True).to(device)
    from models.dinov3_tessera.ops.flash_attention_tessera import TileLinear, DummySchedule
    tl = TileLinear(K, M, bias=True, schedule=DummySchedule(block_m=3, block_k=7)).to(device)
    with torch.no_grad():
        tl.weight.copy_(lin.weight)
        tl.bias.copy_(lin.bias)
    y1 = lin(x)
    y2 = tl(x)
    assert torch.allclose(y1, y2, atol=1e-5), (y1 - y2).abs().max().item()
    print("tilelinear ok")
