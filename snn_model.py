import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNN(nn.Module):
    def __init__(self, n_in=128, n_h=128, n_out=2, beta=0.9):
        super().__init__()
        g = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(n_in, n_h); self.lif1 = snn.Leaky(beta=beta, spike_grad=g)
        self.fc2 = nn.Linear(n_h, n_out); self.lif2 = snn.Leaky(beta=beta, spike_grad=g)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        mem1 = self.lif1.init_leaky(); mem2 = self.lif2.init_leaky()
        out, hid = [], []
        for t in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(self.drop(spk1)), mem2)
            out.append(spk2); hid.append(spk1)
        return torch.stack(out), torch.stack(hid)
