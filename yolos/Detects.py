# Standard Package
from __future__ import annotations
from typing import Any, Tuple, Dict
# FrameWork
import torch
# MyPack!
from yolos.Utils import AP, NMS, IntersectionOverUnion
from yolos.BoundingBox import BoundingBox, BoundingBoxes
from yolos.YoloBoxes import YoloRoot

__all__ = [
    "DetectBox", "DetectBoxes", "Detect", ""
]

# 検出用ボックス
class DetectBox(BoundingBox):
    def __init__(self, labelname: str, labelid: int, xmin: Any, ymin: Any, xmax: Any, ymax: Any, confs: Any) -> None:
        super(DetectBox, self).__init__(labelname, labelid, xmin, ymin, xmax, ymax)
        self.confs = confs
        self.Values.append(confs) # = list([labelname, int(labelid), xmin, ymin, xmax, ymax, confs])
        self.FormatInt = self.FormatInt + ", ( Confidence: {:04.02f}% )"
        self.FormatFloat = self.FormatFloat + ", ( Confidence: {:04.02f}% )"

# 検出用ボックスの格納
class DetectBoxes(BoundingBoxes):
    def __init__(self, width: int, height: int) -> None:
        super(DetectBoxes, self).__init__(width=width, height=height)
        self.Width, self.Height = width, height

# 検出器
class Detect(YoloRoot):
    def __init__(self, Width: int=1, Height: int=1) -> None:
        self.WidthHeight = torch.Tensor([Width, Height])
        
    def __call__(self, Prediction: torch.Tensor, Target: torch.Tensor=None) -> Tuple[Dict, Dict]:
        def _Call_(Decoder: torch.Tensor):
            Result = DetectBoxes(*self.WidthHeight.long().tolist())
            for ClassesLabel in range(self.C):
                Mask = (Decoder[..., -1] == ClassesLabel).unsqueeze(-1).expand_as(Decoder)
                if Mask.numel() == 0: continue
                BBoxes = Decoder[Mask].reshape(-1, 7)
                Boxes = BBoxes[..., [0, 1, 2, 3]].reshape(-1, 4)
                Confs = BBoxes[..., 4].reshape(-1)
                LabelScores = BBoxes[..., 5].reshape(-1)
                keep, count = NMS(Boxes, Confs, self.NMSThreshold)
                Boxes = Boxes[keep].reshape(-1, 4)
                Probs = (Confs[keep] * LabelScores[keep]).reshape(-1, 1)
                result = torch.concat((Boxes, Probs), dim=-1)
                for res in result:
                    args = list(map(lambda x: x.item(), res))
                    Result += DetectBox("None", ClassesLabel, *args)
            return Result
        if Target is not None: return _Call_(Prediction), _Call_(self.Decoder(Target))
        return _Call_(self.Decoder(Prediction, True))
    
    def Decoder(self, Boxes: torch.Tensor, Bpatter: bool=False):
        """
            Center XYPoint To XYPoint
        """
        assert Boxes.dim() < 4 # 0 ~ 3 Clear!
        S, B, C = self.S, self.B, self.C
        X = torch.arange(7).unsqueeze(-1).expand(S, S)
        Y = torch.arange(7).unsqueeze(-1).expand(S, S).transpose(1, 0)
        Class = Boxes[..., 10:].reshape(S, S, C)
        Conf = Boxes[..., [4, 9]].reshape(S, S, B)
        BBoxes = Boxes[..., [0, 1, 2, 3, 5, 6, 7, 8]].reshape(S, S, B, 4)
        ClassScore, ClassIndex = Class.max(-1)
        maskProb = (Conf * ClassScore.unsqueeze(-1).expand_as(Conf)) > self.ProThreshold
        X = X.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        Y = Y.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        XY = torch.concat((X, Y), dim=-1)
        XYMINMAX = BBoxes[maskProb]
        Conf = Conf[maskProb].unsqueeze(-1)
        ClassIndex = ClassIndex.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        ClassScore = ClassScore.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        bbox = torch.concat((XY, XYMINMAX, Conf, ClassScore, ClassIndex), dim=-1)
        if not Bpatter:
            Result = []
            for b in range(0, bbox.size(0), B):
                box1, box2 = bbox[b], bbox[b+1]
                if torch.sum((box1 == box2).long()) == 9: 
                    Result.append(box1)
                else:
                    Result.append(box1)
                    Result.append(box2)
            bbox = torch.vstack(Result)
        XYMIN = ((bbox[..., [0, 1]] + bbox[..., [2, 3]]) / S - .5 * bbox[..., [4, 5]]).reshape(-1, 2) * self.WidthHeight 
        XYMAX = ((bbox[..., [0, 1]] + bbox[..., [2, 3]]) / S + .5 * bbox[..., [4, 5]]).reshape(-1, 2) * self.WidthHeight
        return torch.concat((XYMIN, XYMAX, bbox[..., 6:].reshape(-1, 3)), dim=-1)
    
    def MeanAP(self, Prediction: torch.Tensor, Target: torch.Tensor):
        def OnBatch(P, T):
            P, T = self.__call__(P, T)
            return MeanAP(self.C, self.IoUThreshold)(P, T)
        return torch.mean(torch.Tensor([OnBatch(P, T) for P, T in zip(Prediction, Target)])).item()

def MeanAP(ClassNum: int, iouThreshold: float=0.5):
    def _Call_(Prediction: DetectBoxes, Target: DetectBoxes):
        Prediction.ClassIDSort()
        Target.ClassIDSort()
        APs = torch.zeros(ClassNum)
        for key in range(ClassNum):
            P = DetectBoxes(1, 1)
            for box in Prediction:
                 if box.labelid == key:
                      P += box
            T = DetectBoxes(1, 1)
            for box in Target:
                if box.labelid == key:
                    T += box
            PSize, TSize = len(P), len(T)
            if PSize == 0 and TSize == 0:
                 APs[key] = 1.
                 continue
            elif PSize == 0 or TSize == 0:
                continue

            PBox = torch.Tensor([[b.xmin, b.ymin, b.xmax, b.ymax] for b in P])
            TBox = torch.Tensor([[b.xmin, b.ymin, b.xmax, b.ymax] for b in T])

            iou = IntersectionOverUnion(TBox, PBox).reshape(-1)
            Correct = torch.where(iou >= iouThreshold, 1., 0.)
            Scores = torch.Tensor([[b.confs] for b in P])
            Scores = Scores.tile(Scores.size(0), 1).reshape(-1)
            APs[key] = AP(Scores, Correct)
        return torch.mean(torch.Tensor(APs)).item()
    return _Call_
    
if __name__ == "__main__":
    from Utils import CreateBoxes
    torch.manual_seed(123)
    P = CreateBoxes(3)
    T = CreateBoxes(2)
    print(P)
    print(T)
    iou = IntersectionOverUnion(T, P)
    print(iou)

    detect1 = DetectBoxes(100, 100)
    detect2 = DetectBoxes(100, 100)
    for box in P:
        box = DetectBox("none", *list(map(lambda x: x.item(), box)), confs=torch.rand(1).item())
        detect1 += box
    for box in T:
        detect2 += DetectBox("none", *list(map(lambda x: x.item(), box)), confs=torch.rand(1).item())
    print(detect1)
    print(detect2)

    MAPs = MeanAP(3)
    print(MAPs(detect1, detect2))
