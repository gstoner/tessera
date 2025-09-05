// Tiny 2‑layer Nemotron‑H graph (toy) for Tessera Graph‑IR
// Layer0: Mamba2, Layer1: Attention (GQA), then MLP
// This is illustrative — adapt to your dialect names.

module {
  func.func @nemotron2_demo(%ids: tensor<?xi32>) -> tensor<?x5120xf32> {
    %emb = "tessera.graph.embed"(%ids) {vocab = 131072, hidden = 5120} : (tensor<?xi32>) -> tensor<?x5120xf32>
    %m0  = "tessera.graph.mamba2"(%emb) {m_heads=128, m_head_dim=80, ssm_state=128, chunk=128} : (tensor<?x5120xf32>) -> tensor<?x5120xf32>
    %a1  = "tessera.graph.attn_gqa"(%m0) {heads=40, kv_heads=8, head_dim=128} : (tensor<?x5120xf32>) -> tensor<?x5120xf32>
    %f2  = "tessera.graph.mlp_relu2"(%a1) {ff=20480} : (tensor<?x5120xf32>) -> tensor<?x5120xf32>
    %out = "tessera.graph.rmsnorm"(%f2) {eps=1.0e-5} : (tensor<?x5120xf32>) -> tensor<?x5120xf32>
    return %out : tensor<?x5120xf32>
  }
}
