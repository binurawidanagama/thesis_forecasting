import torch
import torch.nn as nn

class RandomFeatureELM(nn.Module):
    """
    Random features + ridge closed-form.
    Fit: H = phi(X W + b), beta = (H^T H + λI)^{-1} H^T Y
    Predict: Yhat = H beta
    """
    def __init__(self, in_dim, out_dim, hidden=1024, act="tanh", ridge_lambda=1e-2, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # normalize device and dtype handling so tensor/device/dtype mismatches don't occur later
        self.device = torch.device(device)
        self.dtype = torch.get_default_dtype()
        # create parameters directly on the target device/dtype
        self.W = nn.Parameter(torch.randn(in_dim, hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5, requires_grad=False)
        self.act = act
        # keep ridge as a python float; when creating identity matrix we will use H's dtype/device
        self.ridge = float(ridge_lambda)
        self.beta = None
        self.out_dim = out_dim

    def _phi(self, XH):
        if self.act == "tanh":
            return torch.tanh(XH)
        elif self.act == "relu":
            return torch.relu(XH)
        else:
            return XH

    def fit(self, X, Y):
        # X: [N, in_dim], Y: [N, out_dim]
        # coerce X/Y to the same device and dtype as the model parameters
        X = X.to(device=self.device, dtype=self.W.dtype)
        Y = Y.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)    # [N, hidden]
        # ridge solve
        HtH = H.T @ H
        # ensure integer size and matching dtype/device for the identity matrix
        n = int(HtH.size(0))
        lamI = self.ridge * torch.eye(n, device=H.device, dtype=H.dtype)
        self.beta = torch.linalg.solve(HtH + lamI, H.T @ Y)  # [hidden, out_dim]

    def predict(self, X):
        X = X.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)
        return H @ self.beta

