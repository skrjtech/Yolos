import torch
import numpy as np

def EncoderBBox(BBox: torch.Tensor, Width: int, Height:int, S: int=7):

    if isinstance(BBox, list): BBox = torch.Tensor(BBox)
    elif isinstance(BBox, tuple): BBox = torch.Tensor(BBox)
    elif isinstance(BBox, np.ndarray): BBox = torch.Tensor(BBox)
    if BBox.dim() == 1: BBox = BBox.reshape(1, -1)

    S = float(S)
    Label = BBox[..., -1].unsqueeze(-1)
    WH = torch.Tensor([Width, Height]).unsqueeze(0)
    
    XYXY = BBox[..., :4] / torch.cat((WH, WH), dim=1)
    XYC = (XYXY[..., [2, 3]] + XYXY[..., [0, 1]]) / 2.
    WH = (XYXY[..., [2, 3]] - XYXY[..., [0, 1]])

    XYI = (XYC * S).ceil() - 1.
    XYN = (XYC - (XYI / S)) * S

    return torch.cat((XYI, XYN, WH, Label), dim=1)

def DecoderBBox(BBox: torch.Tensor, Width: int, Height:int, S: int=7):

    if isinstance(BBox, list): BBox = torch.Tensor(BBox)
    elif isinstance(BBox, tuple): BBox = torch.Tensor(BBox)
    elif isinstance(BBox, np.ndarray): BBox = torch.Tensor(BBox)
    if BBox.dim() == 1: BBox = BBox.reshape(1, -1)

    S = float(S)
    Label = BBox[..., -1].unsqueeze(-1)
    WH = torch.Tensor([Width, Height]).unsqueeze(0)

    XY0 = BBox[..., [0, 1]] / S
    XYN = BBox[..., [2, 3]] / S + XY0
    XYMIN = (XYN - 0.5 * BBox[..., [4, 5]]) * WH
    XYMAX = (XYN + 0.5 * BBox[..., [4, 5]]) * WH
    
    return torch.cat((XYMIN, XYMAX), dim=1).ceil()

def MakeTargetBBox(BBox: torch.Tensor, S: int, B: int, C: int):

    if isinstance(BBox, list): BBox = torch.Tensor(BBox)
    elif isinstance(BBox, tuple): BBox = torch.Tensor(BBox)
    elif isinstance(BBox, np.ndarray): BBox = torch.Tensor(BBox)
    if BBox.dim() == 1: BBox = BBox.reshape(1, -1)

    N = B * 5 + C
    Label = BBox[..., -1].long()
    Target = torch.zeros(S, S, N)
    X, Y = BBox[..., 0].long(), BBox[..., 1].long()
    XYWH = BBox[..., [2, 3, 4, 5]]
    Target[Y, X, [0, 1, 2, 3, 5, 6, 7, 8]] = torch.cat((XYWH, XYWH), dim=1)
    Target[Y, X, [4, 9]] = torch.Tensor([1., 1.])
    Target[Y, X, B * 5 + Label] = torch.Tensor([1.])
    
    return Target

def NonMaximumSuppression(BBox: torch.Tensor):
    """"""
    pass