# Standard Package
from __future__ import annotations
from typing import Any, Tuple
# FrameWork
import torch
# MyPack!
from yolos.BoundingBox import BoundingBoxes, BoundingBoxCenter, BoundingBox

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

    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str:
        return f"( S: {self.S:3d} | B: {self.B:3d} | C: {self.C:3d} | N[B・5+C]: {self.N:3d} ) | ( ProThreshold: {self.ProThreshold:08.03f} | NMSThreshold: {self.NMSThreshold:08.03f} | IoUThreshold: {self.IoUThreshold:08.03f} ) | ( LambdaObj: {self.LambdaObj:08.03f} | LambdaNoObj: {self.LambdaNoObj:08.03f} )"

class YoloBox(BoundingBox):
    def __call__(self) -> Tuple:
        return (self.labelname, self.labelid, self.xmin, self.ymin, self.xmax, self.ymax)

class YoloBoxCenter(BoundingBoxCenter):
    def __init__(self, labelname: str, labelid: Any, xi: Any, yi: Any, xcenter: Any, ycenter: Any, width: Any, height: Any) -> None:
        super().__init__(labelname, labelid, xcenter, ycenter, width, height)
        self.xi, self.yi = xi, yi
        self.Values = list([labelname, labelid, xi, yi, xcenter, ycenter, width, height])
        self.FormatInt = "( objname: {} | objid: {:5d} ), ( xi: {:3d} | yi: {:3d} ), ( x: {:3d} | y: {:3d} ), ( w: {:3d} | h: {:3d} )"
        self.FormatFloat = "( objname: {} | objid: {:5d} ), ( xi: {:3d} | yi: {:3d} ), ( x: {:08.03f} | y: {:08.03f} ), ( w: {:08.03f} | h: {:08.03f} )"
    
    def ToInt(self) -> BoundingBoxCenter:
        self.Values[1: 7+1] = list(map(int, self.Values[1: 7+1]))
        return self

class YoloBoxesBase(BoundingBoxes, YoloRoot):
    def __init__(self, width: int, height: int, **kwargs) -> None:
        BoundingBoxes.__init__(self, width=width, height=height)
        YoloRoot.__init__(self, **kwargs)

class YoloBoxes(YoloBoxesBase):
    _EncDecFlag = False
    def __init__(self, width: int=1, height: int=1, **kwargs) -> None:
        super(YoloBoxes, self).__init__(width, height, **kwargs)
    
    def __call__(self) -> torch.Tensor: return self.ToTarget()
    def Encoder(self) -> YoloBoxes:
        
        if self._EncDecFlag: return self
        self._EncDecFlag = True
        
        S = self.S
        self.Normalize().ToCenter()
        for idx, box in enumerate(self.Boxes):
            if not isinstance(box, BoundingBoxCenter):
                raise Exception("not BoundingBoxCenter")
            name, id, cx, cy, w, h = box()
            cxi = int(cx * S); cxn = (cx - (cxi / S)) * S
            cyi = int(cy * S); cyn = (cy - (cyi / S)) * S
            self.__setitem__(idx, YoloBoxCenter(name, id, cxi, cyi, cxn, cyn, w, h))
        return self

    def Decoder(self) -> YoloBoxes:

        if not self._EncDecFlag: return self
        self._EncDecFlag = False        
        
        if len(self.Boxes) == 0:
            raise Exception("エンコードされているBoxesがありません")
        
        S = self.S
        for idx, box in enumerate(self.Boxes):
            name, id, cxi, cyi, cxn, cyn, w, h = box()
            xn = (cxn + cxi) / S
            yn = (cyn + cyi) / S
            xmin = (xn - 0.5 * w) 
            ymin = (yn - 0.5 * h)
            xmax = (xn + 0.5 * w)
            ymax = (yn + 0.5 * h)
            self.__setitem__(idx, BoundingBox(name, id, xmin, ymin, xmax, ymax))
        self.ToPoint().DNormalize()
        return self

    def ToTarget(self) -> torch.Tensor:
        self.Encoder()
        S, B, N = self.S, self.B, self.N
        Target = torch.zeros(S, S, N)
        for box in self.Boxes:
            if not isinstance(box, YoloBoxCenter):
                raise Exception("Decodeされていません")
            (name, id, cxi, cyi, cxn, cyn, w, h) = box()
            Target[cxi, cyi, [0, 1, 2, 3, 5, 6, 7, 8]] = torch.Tensor([cxn, cyn ,w, h, cxn, cyn ,w, h])
            Target[cxi, cyi, [4, 9]] = torch.Tensor([1., 1.])
            Target[cxi, cyi, B * 5 + id] = torch.Tensor([1.])
        return Target
    
if __name__ == "__main__":
    BBoxes = YoloBoxes(300, 229)
    BBoxes += YoloBox("apple", 0, 12, 22, 105, 111)
    BBoxes += YoloBox("apple", 0, 71, 60, 175, 164)
    BBoxes += YoloBox("apple", 0, 134, 23, 243, 115)
    BBoxes += YoloBox("apple", 0, 107, 126, 216, 229)
    BBoxes += YoloBox("apple", 0, 207, 138, 298, 229)
    print(BBoxes)