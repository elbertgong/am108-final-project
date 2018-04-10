import torch
import time
import math
import torch
from torch.autograd import Variable

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    # es = s / (percent)
    # rs = es - s
    return '%s' % (asMinutes(s))

def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")

# source: https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def lstm_hidden(laydir,bs,hiddensz):
    return (Variable(torch.zeros(laydir,bs,hiddensz).cuda()), Variable(torch.zeros(laydir,bs,hiddensz).cuda()))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def unpackage_hidden(h):
    """Unwraps hidden states into Tensors."""
    if type(h) == Variable:
        return h.data
    else:
        return tuple(unpackage_hidden(v) for v in h)

def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
