import torch, torch.nn as nn, torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
                                  nn.Conv2d(16,32,3,padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(32, k)
    def forward(self,x):
        a = self.conv(x)
        z = self.pool(a).flatten(1)
        return self.fc(z), a

def grad_cam(model, x, target_idx):
    model.zero_grad()
    logits, acts = model(x)
    score = logits[:, target_idx].sum()
    grads = torch.autograd.grad(score, acts)[0]
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights*acts).sum(1, keepdim=True)
    cam = F.relu(cam); cam = cam / (cam.amax(dim=(2,3), keepdim=True)+1e-8)
    return cam

if __name__ == "__main__":
    torch.manual_seed(0)
    m = SmallCNN(k=5)
    x = torch.randn(2,1,28,28)
    cam = grad_cam(m, x, target_idx=3)
    print("Grad-CAM map shape:", cam.shape)
