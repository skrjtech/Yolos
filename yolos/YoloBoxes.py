from __future__ import annotations
from dataclasses import dataclass
from typing import IO, Any, Union, Tuple, List, Dict

import torch
from BoundingBox import BoundingBoxes, BoundingBoxCenter, BoundingBox

class YoloRoot:
    S: int=7
    B: int=2
    C: int=20
    ProThreshold: float=0.3
    NMSThreshold: float=0.5
    IoUThreshold: float=0.5
    LambdaObj: float=5.
    LambdaNoObj: float=.5
    N: int = None
    def __init__(self, S=None, B=None, C=None, ProThreshold=None, NMSThreshold=None, IoUThreshold=None, LambdaObj=None, LambdaNoObj=None):

        if S is not None: YoloRoot.S = S
        if B is not None: YoloRoot.B = B
        if C is not None: YoloRoot.C = C
        if ProThreshold is not None: YoloRoot.ProThreshold = ProThreshold
        if NMSThreshold is not None: YoloRoot.NMSThreshold = NMSThreshold
        if IoUThreshold is not None: YoloRoot.IoUThreshold = IoUThreshold
        if LambdaObj is not None: YoloRoot.LambdaObj = LambdaObj
        if LambdaNoObj is not None: YoloRoot.LambdaNoObj = LambdaNoObj
        YoloRoot.N = YoloRoot.B * 5 + YoloRoot.C

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"(S: {self.S: 3d} | B: {self.B: 3d} | C: {self.C: 3d} | N[B・5＋C]: {self.N: 3d}) | (ProThreshold: {self.ProThreshold:^.3f} | NMSThreshold: {self.NMSThreshold:^.3f} | IoUThreshold: {self.IoUThreshold:^.3f}) | (LambdaObj: {self.LambdaObj:^.3f} | LambdaNoObj: {self.LambdaNoObj:^.3f})"
        
    def __call__(self) -> Tuple:
        return (self.S, self.B, self.C, self.B * 5 + self.C)

@dataclass(frozen=True)
class YoloBox:
    xmin: float; ymin: float; xmax: float; ymax: float
    labelname: str; labelid: int
    def __call__(self) -> Tuple:
        return (self.xmin, self.ymin, self.xmax, self.ymax, self.labelname, self.labelid)
    def __str__(self) -> str:
        return f"(xmin: {self.xmin:^.3f} | ymin: {self.ymin:^.3f}), (xmax: {self.xmax:^.3f} | ymax: {self.ymax:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"

@dataclass(frozen=True)
class YoloBoxCenter:
    xi: float; yi: float
    x: float; y: float; w: float; h: float
    labelname: str; labelid: int
    def __call__(self) -> Tuple:
        return (self.xi, self.yi, self.x, self.y, self.w, self.h, self.labelname, self.labelid)
    def __str__(self) -> str:
        return f"(xi: {self.xi:^ 3} | yi: {self.yi:^ 3}), (x: {self.x:^.3f} | y: {self.y:^.3f}), (w: {self.w:^.3f} | h: {self.h:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"


class YoloBoxesBase(BoundingBoxes, YoloRoot):
    def __init__(self, *args, **kwargs) -> None:

        BoundingBoxes.__init__(self, *args, **kwargs)
        YoloRoot.__init__(self)

class YoloBoxes(YoloBoxesBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._EncDecFlag = False
    
    def __call__(self) -> torch.Tensor:
        return self.ToTarget()
    
    def Encoder(self) -> YoloBoxes:
        if self._EncDecFlag: return self
        self._EncDecFlag = True
        self.ToCenter()
        S = self.S
        for idx, box in enumerate(self.Boxes):
            if not isinstance(box, BoundingBoxCenter):
                raise Exception("not BoundingBoxCenter")
            cx, cy, w, h, name, id = box()
            cxi = int(cx * S)
            cxn = (cx - (cxi / S)) * S
            cyi = int(cy * S)
            cyn = (cy - (cyi / S)) * S
            self.__setitem__(idx, YoloBoxCenter(cxi, cyi, cxn, cyn, w, h, name, id))
        return self

    def Decoder(self) -> YoloBoxes:
        if not self._EncDecFlag: return self
        self._EncDecFlag = False        
        S = self.S
        if len(self.Boxes) == 0:
            raise Exception("エンコードされているBoxesがありません")
        for idx, box in enumerate(self.Boxes):
            (cxi, cyi, cxn, cyn, w, h, name, id) = box()
            xn = (cxn + cxi) / S
            yn = (cyn + cyi) / S
            xmin = (xn - 0.5 * w) 
            ymin = (yn - 0.5 * h)
            xmax = (xn + 0.5 * w)
            ymax = (yn + 0.5 * h)
            self.__setitem__(idx, BoundingBox(xmin, ymin, xmax, ymax, name, id))
        self.ToPoint()
        return self

    def ToTarget(self) -> torch.Tensor:
        self.Encoder()
        S, B, N = self.S, self.B, self.N
        Target = torch.zeros(S, S, N)
        for box in self.Boxes:
            if not isinstance(box, YoloBoxCenter):
                raise Exception("Decodeされていません")
            (cxi, cyi, cxn, cyn, w, h, name, id) = box()
            Target[cxi, cyi, [0, 1, 2, 3, 5, 6, 7, 8]] = torch.Tensor([cxn, cyn ,w, h, cxn, cyn ,w, h])
            Target[cxi, cyi, [4, 9]] = torch.Tensor([1., 1.])
            Target[cxi, cyi, B * 5 + id] = torch.Tensor([1.])
        return Target
    
class Detect(YoloRoot):
    def __init__(self, Width: int=1, Height: int=1) -> None:
        self.WidthHeight = torch.Tensor([Width, Height])
        
    def __call__(self, Prediction: torch.Tensor, Target: torch.Tensor) -> Tuple[Dict, Dict]:
        Prediction, Target = self.Decoder(Prediction, True), self.Decoder(Target)
        def _Call_(Decoder: torch.Tensor):
            Result = {}
            for ClassesLabel in range(self.C):
                Mask = (Decoder[..., -1] == ClassesLabel).unsqueeze(-1).expand_as(Decoder)
                if Mask.numel() == 0: continue
                BBoxes = Decoder[Mask].reshape(-1, 7)
                Boxes = BBoxes[..., [0, 1, 2, 3]].reshape(-1, 4)
                Confs = BBoxes[..., 4].reshape(-1)
                LabelScores = BBoxes[..., 5].reshape(-1)
                keep, count = self.NMS(Boxes, Confs)
                Boxes = Boxes[keep].reshape(-1, 4)
                Probs = (Confs[keep] * LabelScores[keep]).reshape(-1, 1)
                Result[ClassesLabel] = torch.concat((Boxes, Probs), dim=-1)
            return Result
        return _Call_(Prediction), _Call_(Target)
    
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

    def NMS(self, Boxes: torch.Tensor, Scores: torch.Tensor, top_k: int=200) -> Tuple[torch.Tensor, int]:
        count = 0
        keep = Scores.new(Scores.size(0)).zero_().long()
        x1 = Boxes[:, 0]
        y1 = Boxes[:, 1]
        x2 = Boxes[:, 2]
        y2 = Boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        tmp_x1 = Boxes.new()
        tmp_y1 = Boxes.new()
        tmp_x2 = Boxes.new()
        tmp_y2 = Boxes.new()
        tmp_w =  Boxes.new()
        tmp_h =  Boxes.new()
        v, idx = Scores.sort(0)
        idx = idx[-top_k:]
        while idx.numel() > 0:
            i = idx[-1]
            
            keep[count] = i
            count += 1
            
            if idx.size(0) == 1: break
            idx = idx[:-1]
            
            tmp_x1 = torch.index_select(x1, 0, idx)
            tmp_y1 = torch.index_select(y1, 0, idx)
            tmp_x2 = torch.index_select(x2, 0, idx)
            tmp_y2 = torch.index_select(y2, 0, idx)
            
            tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
            tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
            tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
            tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

            tmp_w.resize_as_(tmp_x2)
            tmp_h.resize_as_(tmp_y2)

            tmp_w = tmp_x2 - tmp_x1
            tmp_h = tmp_y2 - tmp_y1

            tmp_w = torch.clamp(tmp_w, min=0.0)
            tmp_h = torch.clamp(tmp_h, min=0.0)

            inter = tmp_w*tmp_h

            rem_areas = torch.index_select(area, 0, idx)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union

            idx = idx[IoU.le(self.NMSThreshold)]
            
        return keep, count

    def AP(self, Scores: torch.Tensor, Correct: torch.Tensor) -> torch.Tensor:
        
        if torch.sum(Correct) == 0:
            return 0.

        IndexSort = torch.sort(Scores, descending=True)[-1] # 降順
        Correct = Correct[IndexSort]

        TP = torch.cumsum(Correct, dim=-1)
        Precision = TP / (torch.arange(TP.size(0)) + 1.)
        Recall = TP / torch.sum(Correct, dim=-1)
        
        # PrecisionFlip = Precision.flip(dims=(0,))
        # PrecisionFlip = torch.cummax(PrecisionFlip, dim=0)[0].flip(dims=(0,))

        Precision = torch.concat([torch.Tensor([0]), Precision, torch.Tensor([0])], dim=-1)
        Recall = torch.concat([torch.Tensor([0]), Recall, torch.Tensor([1])], dim=-1)

        Recall = Recall[1:] - Recall[:-1]
        
        return torch.sum(Recall * Precision[1:], dim=-1)

    def MeanAP(self, Prediction: torch.Tensor, Target: torch.Tensor):
        Prediction, Target = self.__call__(Prediction, Target)
        
        from yolos.Models import IoU as IntersectionOverUnion
        
        APs = torch.zeros(self.C)
        for key in range(self.C):
            PVal, TVal = Prediction[key], Target[key]
            if PVal.size(0) == 0 and TVal.size(0) == 0: 
                APs[key] = 1.
                continue
            elif PVal.size(0) == 0 or TVal.size(0) == 0:
                continue
                
            PBox, TBox = PVal[..., :4], TVal[..., :4]
            iou = IntersectionOverUnion(TBox, PBox).reshape(-1)
            Correct = torch.where(iou >= self.IoUThreshold, 1., 0.)
            APs[key] = self.AP(PVal[..., -1].reshape(-1), Correct)
        return torch.mean(torch.Tensor(APs)).item()
    
if __name__ == "__main__":
    BBoxes = BoundingBoxes(400, 400)
    BBoxes += BoundingBox(100, 100, 250, 250, "banana", 0)
    BBoxes += BoundingBox(50, 50, 100, 100, "banana", 1)
    BBoxes += BoundingBox(245, 245, 360, 360, "banana", 2)

    YoloRoot(C=3)
    yoloboxes = YoloBoxes(width=10, height=10)
    yoloboxes += BoundingBox(1, 1, 5, 5, "orange", 0)
    print(yoloboxes)
    target = yoloboxes()
    print(yoloboxes)

