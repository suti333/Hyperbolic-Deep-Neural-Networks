import torch

def cosh(x, clamp = 15):
    return x.clamp(-clamp, clamp).cosh()

def sinh(x, clamp = 15):
    return x.clamp(-clamp, clamp).sinh()

def tanh(x, clamp = 15):
    return x.clamp(-clamp, clamp).tanh()

def arcosh(x):
    return Arcosh.apply(x)

def arsinh(x):
    return Arsinh.apply(x)

def artanh(x):
    return Artanh.apply(x)


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp_min(1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        res = torch.log(z + torch.sqrt(z.pow(2) - 1.0))
        return res.to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output / torch.sqrt(input.pow(2) - 1)
        return grad_input
    
class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        res = torch.log((z + torch.sqrt(1 + z.pow(2))).clamp_min(1e-15))
        return res.to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output / torch.sqrt(1 + input.pow(2))
        return grad_input
    
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-6, 1 - 1e-6)
        ctx.save_for_backward(x)
        z = x.double()
        res = (torch.log(1 + z).sub(torch.log(1 - z))).mul(0.5)
        return res.to(x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output / (1 - input.pow(2))
        return grad_input
