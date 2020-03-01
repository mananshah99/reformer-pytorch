import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g, h = None):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.h = None if h is None else Deterministic(h)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=True) if self.h is None else x1 + self.f(x2, record_rng=True) + self.h(torch.cat([x1, x2], dim=-1), record_rng=True)
            y2 = x2 + self.g(y1, record_rng=True) if self.h is None else x2 + self.g(y1, record_rng=True)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy):
        if self.h is None: return self.backward_pass_original(y, dy)
        else: return self.backward_pass_dev(y, dy)
    
    def backward_pass_original(self, y, dy):
        # (y_1, y_2)
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        #(y_1bar, y_2bar)
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx

    def backward_pass_dev(self, y, dy):
        # (y_1, y_2)
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        #(y_1bar, y_2bar)
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

        with torch.enable_grad():
            x2.requires_grad = True

            ### update to account for h
            hx2 = self.h(torch.cat([x1, x2], dim=-1), set_rng=True)
            torch.autograd.backward(hx2, dx1, retain_graph=True)

        with torch.no_grad():

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, recurrence):
        # currently, the recurrence parameter is 
        for block in blocks:
            x = block(x)

        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy)
        return dy, None, None

class ReversibleSequence(nn.Module):
    def __init__(self, blocks, recurrence = False, layer_dropout = 0.):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g, h=h) for f, g, h in blocks])
        self.recurrence = recurrence

    def forward(self, x):
        blocks = self.blocks

        if self.training and self.layer_dropout > 0:
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks

        return _ReversibleFunction.apply(x, blocks, self.recurrence)
