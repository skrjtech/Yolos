from __future__ import annotations
from typing import Any, Union, List, Tuple
import numpy as np

__all__ = [
    "BoundingBox",
    "BoundingBoxCenter",
    "BoundingBoxes"
]

class BoundingBox(object):
    """
    座標とオブジェクト属性を格納する.\n
    Xmin Ymin Xmax Ymax\n
    LabelName Labelid\n
    """

    def __init__(self, xmin: Union[str, int, float], ymin: Union[str, int, float], xmax: Union[str, int, float], ymax: Union[str, int, float], labelname: str, labelid: Union[str, int]) -> None:
        if isinstance(xmin, str) or isinstance(xmin, int): xmin = float(xmin)
        if isinstance(ymin, str) or isinstance(ymin, int): ymin = float(ymin)
        if isinstance(xmax, str) or isinstance(xmax, int): xmax = float(xmax)
        if isinstance(ymax, str) or isinstance(ymax, int): ymax = float(ymax)
        assert xmin < xmax, xmax > xmin 
        assert ymin < ymax, ymax > ymin 
        self.xmin, self.ymin, self.xmax, self.ymax, self.labelname, self.labelid = xmin, ymin, xmax, ymax, labelname, labelid

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.xmin, self.ymin, self.xmax, self.ymax, self.labelname, self.labelid

    def __str__(self) -> str:
        return f"(xmin: {self.xmin:^.3f} | ymin: {self.ymin:^.3f}), (xmax: {self.xmax:^.3f} | ymax: {self.ymax:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
    def Int(self):
        self.xmin, self.ymin, self.xmax, self.ymax = tuple(map(int, [self.xmin, self.ymin, self.xmax, self.ymax])) 
    
class BoundingBoxCenter:
    def __init__(self, xcenter: Union[int, float], ycenter: Union[int, float], width: Union[int, float], height: Union[int, float], labelname: str, labelid: Union[str, int]) -> None:
        self.xcenter, self.ycenter, self.width, self.height, self.labelname, self.labelid = xcenter, ycenter, width, height, labelname, labelid
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.xcenter, self.ycenter, self.width, self.height, self.labelname, self.labelid

    def __str__(self) -> str:
        return f"(xcenter: {self.xcenter:^.3f} | ycenter: {self.ycenter:^.3f}), (width: {self.width:^.3f} | height: {self.height:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
    def Int(self):
        self.xcenter, self.ycenter, self.width, self.height = tuple(map(int, [self.xcenter, self.ycenter, self.width, self.height])) 

class BaseBoxes:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.Box: List[Union[BoundingBox, BoundingBoxCenter]] = list()
        self.labelnamelenght = 0

    def __call__(self) -> List[Union[BoundingBox, BoundingBoxCenter]]:
        return self.Box

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self) -> Any:
        if self.idx == len(self.Box):
            raise StopIteration()
        ret = self.Box[self.idx]
        self.idx += 1
        return ret
    
    def __str__(self) -> str:
        for bbox in self.Box: 
            if self.labelnamelenght < len(bbox.labelname): self.labelnamelenght = len(bbox.labelname)
        output = f"Width({self.width:^ 5}), Height({self.height:^ 5})\n"
        for idx, b in enumerate(self.Box):
            b = str(b).replace(b.labelname, b.labelname.center(self.labelnamelenght))
            output += f"  ({idx:^ 3}){b}\n"
        return output

    def __iadd__(self, bbox: Union[BoundingBox, BoundingBoxCenter]) -> BoundingBoxes:
        # if self.labelnamelenght < len(bbox.labelname): self.labelnamelenght = len(bbox.labelname)
        self.Box += [bbox]
        return self

    def __len__(self) -> int:
        return len(self.Box)

    def __setitem__(self, idx: int, bbox: Union[BoundingBox, BoundingBoxCenter]) -> None:
        # if self.labelnamelenght < len(bbox.labelname): self.labelnamelenght = len(bbox.labelname)
        self.Box[idx] = bbox
        return None

    def __getitem__(self, idx: int) -> Union[BoundingBox, BoundingBoxCenter]:
        return self.Box[idx]

    def __dellitem__(self, idx: int) -> None:
        del self.Box[idx]
        return None
    
def PixelRepair(pixel):
    if (pixel - int(pixel)) < 0.5: return int(pixel)
    else: return int(pixel) + 1

class BoundingBoxes(BaseBoxes):
    """
    BoundingBoxオブジェクトの格納する.
    PointToCenter 座標の変換
    CenterToPoint 中心座標の変換
    """

    #BoxList: List[BoundingBox] = list()
    def __init__(self, width: int, height: int) -> None:
        super(BoundingBoxes, self).__init__(width, height)
        self.width, self.height = width, height
        self._NormFlag = False
        self._CenterFlag = False

    def Size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def append(self, bbox: Union[BoundingBox, BoundingBoxCenter]) -> None:
        if self.labelnamelenght < len(bbox.labelname): self.labelnamelenght = len(bbox.labelname)
        self.Box += [bbox]
        return None
    
    def ToCenter(self) -> BoundingBoxes:
        if self._CenterFlag: return self
        self._CenterFlag = True
        for idx, bbox in enumerate(self.Box):
            if not isinstance(bbox, BoundingBox): continue
            xmin, ymin, xmax, ymax, labelname, labelid = bbox()
            self.Box[idx] = BoundingBoxCenter((xmax + xmin) * .5, (ymax + ymin) * .5, (xmax - xmin), (ymax - ymin), labelname, labelid)
        return self

    def ToPoint(self) -> BoundingBoxes:
        if not self._CenterFlag: return self
        self._CenterFlag = False
        for idx, bbox in enumerate(self.Box):
            if not isinstance(bbox, BoundingBoxCenter): continue
            cx, cy, w, h, labelname, labelid = bbox()
            (w, h) = (w * .5, h * .5)
            self.Box[idx] = BoundingBox(cx - w, cy - h, cx + w, cy + h, labelname, labelid)
        return self

    def Normalize(self) -> BoundingBoxes:
        if self._NormFlag: return self
        self._NormFlag = True
        for i, b in enumerate(self.Box):
            A, B, C, D, L1, L2 = b()
            A /= self.width
            B /= self.height
            C /= self.width
            D /= self.height
            if isinstance(b, BoundingBox): box = BoundingBox(A, B, C, D, L1, L2)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(A, B, C, D, L1, L2)
            self.__setitem__(i, box)
        return self

    def DNormalize(self) -> BoundingBoxes:
        if not self._NormFlag: return self
        self._NormFlag = False
        for i, b in enumerate(self.Box):
            A, B, C, D, L1, L2 = b()
            A = PixelRepair(A * self.width)
            B = PixelRepair(B * self.height)
            C = PixelRepair(C * self.width)
            D = PixelRepair(D * self.height)
            if isinstance(b, BoundingBox): box = BoundingBox(A, B, C, D, L1, L2)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(A, B, C, D, L1, L2)
            self.__setitem__(i, box)
        return self

    def Int(self) -> BoundingBoxes:
        for b in self.Box: b.Int()
        return self
    
    def ClassIDSort(self) -> BoundingBoxes:
        index = list(np.argsort([box.labelid for box in self.Box]))
        self.Box = list(np.array(self.Box)[index])
        return self
    
    def CallID(self, id) -> BoundingBoxes:
        self.ClassIDSort()
        Boxes = BoundingBoxes(self.width, self.height)
        for Box in self.Box:
            if Box.labelid == id:
                Boxes += Box
        return Boxes

if __name__ == "__main__":
    bboxlist = BoundingBoxes(10, 10)
    bboxlist += BoundingBox(1, 2, 10, 10, "apple", 1)
    bboxlist += BoundingBox(2, 2, 10, 10, "orange", 0)
    bboxlist += BoundingBox(2, 9, 10, 10, "banana", 2)

    print(bboxlist)
    print(bboxlist.Normalize())
    print(bboxlist.Normalize())
    print(bboxlist.DNormalize())
    print(bboxlist.DNormalize())

    print(bboxlist.ToCenter())
    print(bboxlist.ToCenter())
    print(bboxlist.ToPoint())
    print(bboxlist.ToPoint())
    