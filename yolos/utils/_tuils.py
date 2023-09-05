import torch
import numpy as np

def TorchIsistance(Tensor: any) -> torch.Tensor:

    T = lambda: torch.Tensor(Tensor)
    if isinstance(Tensor, list): return T()
    elif isinstance(Tensor, tuple): return T()
    elif isinstance(Tensor, np.ndarray): return T()
    return Tensor


def IntersectionOverUnion(BBoxP: torch.Tensor, BBoxT: torch.Tensor) -> torch.Tensor:

    BBoxP = TorchIsistance(BBoxP)
    BBoxT = TorchIsistance(BBoxT)

    if (BBoxP.dim() == 1): BBoxP = BBoxP.unsqueeze(0)
    if (BBoxT.dim() == 1): BBoxT = BBoxT.unsqueeze(0)

    N = BBoxP.size(0)
    M = BBoxT.size(0)

    XYMIN = torch.max(
        BBoxP[..., :2].unsqueeze(1).expand(N, M, 2),
        BBoxT[..., :2].unsqueeze(0).expand(N, M, 2),
    )
    XYMAX = torch.min(
        BBoxP[..., 2:].unsqueeze(1).expand(N, M, 2),
        BBoxT[..., 2:].unsqueeze(0).expand(N, M, 2),
    )

    WH = torch.clamp(XYMAX - XYMIN, min=0)
    Intersection = WH[..., 0] * WH[..., 1]

    Area1 = (BBoxP[..., 2] - BBoxP[..., 0]) * (BBoxP[..., 3] - BBoxP[..., 1])
    Area2 = (BBoxT[..., 2] - BBoxT[..., 0]) * (BBoxT[..., 3] - BBoxT[..., 1])
    Area1 = Area1.unsqueeze(1).expand_as(Intersection)
    Area2 = Area2.unsqueeze(0).expand_as(Intersection)

    Union = Area1 + Area2 - Intersection

    return Intersection / Union

def EncoderBBox(BBox: torch.Tensor, Width: int, Height: int, S: int = 7) -> torch.Tensor:

    """
        BBox: [[Xmin, Ymin, Xmax, Ymax, Label],...]
        Width:
        Height:
        S:
        return [[XIndex, YIndex, CenterX, CenterY, Width, Height],...]
    """

    BBox = TorchIsistance(BBox)
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

def DecoderBBox(BBox: torch.Tensor, Width: int, Height: int, S: int = 7) -> torch.Tensor:

    """
        BBox: [[XIndex, YIndex, CenterX, CenterY, Width, Height],...]
        Width:
        Height:
        S:
        return [[Xmin, Ymin, Xmax, Ymax, Label],...]
    """

    BBox = TorchIsistance(BBox)
    if BBox.dim() == 1: BBox = BBox.reshape(1, -1)

    S = float(S)
    Label = BBox[..., -1].unsqueeze(-1)
    WH = torch.Tensor([Width, Height]).unsqueeze(0)

    XY0 = BBox[..., [0, 1]] / S
    XYN = BBox[..., [2, 3]] / S + XY0
    XYMIN = (XYN - 0.5 * BBox[..., [4, 5]]) * WH
    XYMAX = (XYN + 0.5 * BBox[..., [4, 5]]) * WH

    return torch.cat((XYMIN, XYMAX, Label), dim=1).ceil()

def MakeTargetBBox(BBox: torch.Tensor, S: int, B: int, C: int) -> torch.Tensor:

    """
        BBox: [[XIndex, YIndex, CenterX, CenterY, Width, Height, Label],...]
        S:
        B:
        C
        return Tensor(7, 7, B * 5 + C)
    """

    BBox = TorchIsistance(BBox)
    if (BBox.dim() == 1): BBox = BBox.reshape(1, -1)

    N = B * 5 + C
    Label = BBox[..., -1].unsqueeze(-1).long()
    Target = torch.zeros(S, S, N)

    X = BBox[..., 0].unsqueeze(-1).long()
    Y = BBox[..., 1].unsqueeze(-1).long()

    XYWH = BBox[..., [2, 3, 4, 5]]
    Target[Y, X, [0, 1, 2, 3, 5, 6, 7, 8]] = torch.cat((XYWH, XYWH), dim=1)
    Target[Y, X, [4, 9]] = torch.Tensor([1., 1.])
    Target[Y, X, B * 5 + Label] = torch.Tensor([1.])

    return Target

def DetectBBox(BBox: torch.Tensor, probThreshold: float, S: int, B: int, C: int) -> torch.Tensor:

    """
        BBox: Tensor(S, S, N)
        probThreshold:
        S:
        B:
        C:
        return [[Xi, Yi, Xmin, Ymin, Xmax, Ymax, Confidence, ClassScore, ClassIndex], ...]
    """

    assert BBox.dim() < 4 # 0 ~ 3 Clear!

    X = torch.arange(7).unsqueeze(-1).expand(S, S)
    Y = torch.arange(7).unsqueeze(-1).expand(S, S).transpose(1, 0)

    Class = BBox[..., 10:].reshape(S, S, C)
    Conf = BBox[..., [4, 9]].reshape(S, S, B)
    BBoxes = BBox[..., [0, 1, 2, 3, 5, 6, 7, 8]].reshape(S, S, B, 4)

    ClassScore, ClassIndex = Class.max(-1)
    maskProb = (Conf * ClassScore.unsqueeze(-1).expand_as(Conf)) > probThreshold

    X = X.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
    Y = Y.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
    XY = torch.concat((X, Y), dim=-1)
    XYMINMAX = BBoxes[maskProb]
    Conf = Conf[maskProb].unsqueeze(-1)
    ClassIndex = ClassIndex.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
    ClassScore = ClassScore.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)

    return torch.concat((XY, XYMINMAX, Conf, ClassScore, ClassIndex), dim=-1)

def NonMaximumSuppression(BBox: torch.Tensor, Scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:

    """
        BBox: [[Xmin, Ymin, Xmax, Ymax], ...]
        Scores: [Score Value, ...]
        threshold:
        return [index, ...]
    """

    BBox = TorchIsistance(BBox)
    if (BBox.size(0) == 0): return BBox
    if (BBox.dim() == 1): BBox = BBox.unsqueeze(0)

    X1, Y1, X2, Y2 = list(map(lambda x: x.squeeze(1), torch.hsplit(BBox, 4)))

    Areas = (X2 - X1 + 1) * (Y2 - Y1 + 1)
    _, Index = torch.sort(Scores, descending=True)
    Output = []
    while Index.numel() > 0:
        Select = Index.long().item() if (Index.numel() == 1) else Index[0].long()

        Output.append(Select)
        if (Index.numel() == 1): break

        # Cul IoU
        index = Index[1:].long()
        x1 = X1[index].clamp(min=X1[Select])
        y1 = Y1[index].clamp(min=Y1[Select])
        x2 = X2[index].clamp(max=X2[Select])
        y2 = Y2[index].clamp(max=Y2[Select])
        Intersection = (torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0))
        Unions = Areas[Select] + Areas[index] - Intersection
        IoUs = Intersection / Unions

        # Delete Index
        IndexKeep = (IoUs <= threshold).nonzero().squeeze()
        if (IndexKeep.numel() == 0): break
        Index = Index[IndexKeep + 1]

    return torch.LongTensor(Output)


def DetectProcessing(Predicts: torch.Tensor, probThreshold: float, iouThreshold: float, Width: int, Height: int, S: int, B: int, C: int):
    Detect = DetectBBox(Predicts, probThreshold, S, B, C)

    Container = []
    for label in range(C):
        Select = Detect[(Detect[..., -1] == label).unsqueeze(-1).expand_as(Detect)].reshape(-1, 9)
        Indexs = Select[..., [0, 1]]
        Boxes = Select[..., [2, 3, 4, 5]]
        Scores = Select[..., -1]
        NMS = NonMaximumSuppression(Boxes, Scores, iouThreshold)
        if (NMS.numel() == 0): continue
        Indexs = Indexs[NMS]
        Boxes = Boxes[NMS]
        Scores = Scores[NMS].unsqueeze(-1)
        Labels = torch.Tensor([label]).unsqueeze(-1).expand(NMS.size(0), 1)
        output = torch.concat((Indexs, Boxes, Scores, Labels), dim=-1)
        Decoder = DecoderBBox(output, Width, Height, S)
        Container += [output]

    return Container


def AveragePrecision(Predict: torch.Tensor, Target: torch.Tensor, C: int):
    output = {}
    for label in range(len(Target)):
        pBBox = Predict[label]
        tBBox = Target[label]

        output[label] = []
        for tbox in tBBox:
            Size = pBBox.size(0)
            _, Sorted = torch.sort(pBBox[..., -1], descending=True)
            pBBox = pBBox[Sorted]
            Scores = pBBox[..., -1].reshape(-1)
            iou = (IntersectionOverUnion(pBBox[..., :4], tbox[:4]) > 0.5).reshape(-1)
            TP = torch.cumsum((iou == True).long(), dim=0)
            FP = torch.cumsum((iou == False).long(), dim=0)
            Precision = TP / (TP + FP)
            Recall = TP / Size
            Precision = torch.concat((torch.Tensor([0]), Precision, torch.Tensor([0])))
            Recall = torch.concat((torch.Tensor([0]), Recall, torch.Tensor([1])))
            reversPrecision = Precision.flip(dims=(0,))
            PrecisionValues = torch.cummax(reversPrecision, dim=0)[0].flip(dims=(0,))
            AP = (torch.diff(Recall) * Precision[1:]).sum()
            output[label] += [AP]

    return output


def MeanAveragePrecision(AP: dict, C: int):
    Output = {key: 0 for key in AP.keys()}
    for key, value in AP.items():
        value = torch.Tensor(value).mean()
        Output[key] += value
    return Output

if __name__ == "__main__":

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    BBox = [
        [100, 100, 300, 300, 0],
        [30, 30, 70, 70, 1],
        [310, 310, 370, 370, 2],
    ]

    Encoder = EncoderBBox(BBox, 400, 400, 7)
    Target = MakeTargetBBox(Encoder, 7, 2, 3)

    TargetDetect = DetectProcessing(Target, 0.2, 0.5, 400, 400, 7, 2, 3)
    PredictDetect = DetectProcessing(torch.rand(7, 7, 13), 0.2, 0.5, 400, 400, 7, 2, 3)

    print(TargetDetect)