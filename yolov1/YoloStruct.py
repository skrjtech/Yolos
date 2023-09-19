from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict

import torch
from BoundingBox import BoundingBoxes, BoundingBoxCenter, BoundingBox

@dataclass(frozen=False)
class YoloRoot:
    S: int=7
    B: int=2
    C: int=20
    ProThreshold: float=0.1
    NMSThreshold: float=0.1
    IoUThreshold: float=0.1
    LambdaObj: float=5.
    LambdaNoObj: float=.5
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

class YoloBaseStruct:
    def __init__(self) -> None:
        self.GridBox = list()
        self.labelnamelenght = 0

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self) -> Any:
        if self.idx == len(self.Box):
            raise StopIteration()
        ret = self.GridBox[self.idx]
        self.idx += 1
        return ret
    
    def __str__(self) -> str:
        for bbox in self.GridBox: 
            if self.labelnamelenght < len(bbox.labelname): self.labelnamelenght = len(bbox.labelname)
        output = ""
        for idx, b in enumerate(self.GridBox):
            b = str(b).replace(b.labelname, b.labelname.center(self.labelnamelenght))
            output += f"({idx:^ 3}){b}\n"
        return output
    
    def __len__(self) -> int:
        return len(self.GridBox)
    
    def __iadd__(self, bbox: Union[YoloBox, YoloBoxCenter]) -> None:
        self.GridBox += [bbox]
        return None
    
    def __setitem__(self, idx: int, bbox: Union[YoloBox, YoloBoxCenter]) -> YoloStruct:
        self.GridBox[idx] = bbox
        return self
    
    def __getitem__(self, idx: int) -> Union[YoloBox, YoloBoxCenter]:
        return self.GridBox[idx]

    def __dellitem__(self, idx: int) -> None:
        del self.GridBox[idx]
        return None
    
    def ToBoundingBoxes(self):
        raise Exception("未設定")
    
class YoloStruct(YoloBaseStruct):
    def __init__(self, Root: YoloRoot, BBoxes: BoundingBoxes) -> None:
        super(YoloStruct, self).__init__()
        self.root = Root
        self.bboxes = BBoxes
    
    def __call__(self) -> Any:
        return self.ToTarget()
    
    def Encoder(self) -> YoloStruct:
        S = self.root.S
        if len(self.GridBox) == 0:
            for box in self.bboxes():
                if not isinstance(box, BoundingBoxCenter):
                    raise Exception("not BoundingBoxCenter")
                cx, cy, w, h, name, id = box()
                cxi = int(cx * S)
                cxn = (cx - (cxi / S)) * S
                cyi = int(cy * S)
                cyn = (cy - (cyi / S)) * S
                self.GridBox.append(YoloBoxCenter(cxi, cyi, cxn, cyn, w, h, name, id))
        else:
            for idx, box in enumerate(self.bboxes()):
                if not isinstance(box, BoundingBoxCenter):
                    raise Exception("not BoundingBoxCenter")
                cx, cy, w, h, name, id = box()
                cxi = int(cx * S)
                cxn = (cx - (cxi / S)) * S
                cyi = int(cy * S)
                cyn = (cy - (cyi / S)) * S
                self.GridBox[idx] = YoloBoxCenter(cxi, cyi, cxn, cyn, w, h, name, id)
        return self

    def Decoder(self) -> YoloStruct:
        S = self.root.S
        if len(self.GridBox) == 0:
            raise Exception("エンコードされているGridBoxがありません")
        for idx, box in enumerate(self.GridBox):
            (cxi, cyi, cxn, cyn, w, h, name, id) = box()
            xn = (cxn + cxi) / S
            yn = (cyn + cyi) / S
            xmin = (xn - 0.5 * w) 
            ymin = (yn - 0.5 * h)
            xmax = (xn + 0.5 * w)
            ymax = (yn + 0.5 * h)
            self.GridBox[idx] = YoloBox(xmin, ymin, xmax, ymax, name, id)
        return self

    def ToTarget(self) -> torch.Tensor:
        S, B, C, N = self.root() 
        Target = torch.zeros(S, S, N)
        for box in self.GridBox:
            (cxi, cyi, cxn, cyn, w, h, name, id) = box()
            Target[cxi, cyi, [0, 1, 2, 3, 5, 6, 7, 8]] = torch.Tensor([cxn, cyn ,w, h, cxn, cyn ,w, h])
            Target[cxi, cyi, [4, 9]] = torch.Tensor([1., 1.])
            Target[cxi, cyi, B * 5 + id] = torch.Tensor([1.])
        return Target
    
class Detect:
    def __init__(self, yoloStruct: YoloStruct) -> None:
        self.root = yoloStruct.root
        self.yoloStruct = yoloStruct
        self.S, self.B, self.C, self.N = self.root()
        self.Result = None

    def __call__(self, Prediction: torch.Tensor) -> Dict[int:torch.Tensor]:
        Decoder = self.Decoder(Prediction)
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
        self.Result = Result
        return Result
    
    def Decoder(self, PredictBox: torch.Tensor, Bpatter: bool=False):
        assert PredictBox.dim() < 4 # 0 ~ 3 Clear!
        self.PredictBox = PredictBox
        S, B, C = 7, 2, 3
        X = torch.arange(7).unsqueeze(-1).expand(S, S)
        Y = torch.arange(7).unsqueeze(-1).expand(S, S).transpose(1, 0)
        Class = PredictBox[..., 10:].reshape(S, S, C)
        Conf = PredictBox[..., [4, 9]].reshape(S, S, B)
        BBoxes = PredictBox[..., [0, 1, 2, 3, 5, 6, 7, 8]].reshape(S, S, B, 4)
        ClassScore, ClassIndex = Class.max(-1)
        maskProb = (Conf * ClassScore.unsqueeze(-1).expand_as(Conf)) > self.root.ProThreshold
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
        
        XY0 = bbox[..., [0, 1]]
        XYMIN = ((bbox[..., [0, 1]] + bbox[..., [2, 3]]) / S - .5 * bbox[..., [4, 5]]).reshape(-1, 2)
        XYMAX = ((bbox[..., [0, 1]] + bbox[..., [2, 3]]) / S + .5 * bbox[..., [4, 5]]).reshape(-1, 2)
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

            idx = idx[IoU.le(self.root.IoUThreshold)]
            
        return keep, count

    def AP(self):
        return

    def MeanAP(self):
        return
    
if __name__ == "__main__":
    BBoxes = BoundingBoxes(400, 400)
    BBoxes += BoundingBox(100, 100, 250, 250, "banana", 0)
    BBoxes += BoundingBox(50, 50, 100, 100, "banana", 1)
    BBoxes += BoundingBox(245, 245, 360, 360, "banana", 2)

    # Step1
    BBoxes.Normalize()
    print(BBoxes)
    # Step2
    BBoxes.ToCenter()
    print(BBoxes)
    # Step3
    yoloStruct = YoloStruct(YoloRoot(C=3), BBoxes)
    print(yoloStruct.Encoder())
    ret = yoloStruct()
    # Step4
    print(BBoxes.ToPoint())
    # Step5
    print(BBoxes.DNormalize())
    # Step6
    print(yoloStruct.Decoder())
    # Step7
    detect = Detect(yoloStruct)
    print(detect(ret))
    