from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict

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

    def __call__(self) -> Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], str, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax, self.labelname, self.labelid
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        if isinstance(self.xmin, float):
            return f"(xmin: {self.xmin:^.3f} | ymin: {self.ymin:^.3f}), (xmax: {self.xmax:^.3f} | ymax: {self.ymax:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
        return f"(xmin: {self.xmin: 4d} | ymin: {self.ymin: 4d}), (xmax: {self.xmax: 4d} | ymax: {self.ymax: 4d}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
    def ToInt(self) -> BoundingBox:
        self.xmin, self.ymin, self.xmax, self.ymax = tuple(map(int, [self.xmin, self.ymin, self.xmax, self.ymax]))
        return self
    
class BoundingBoxCenter:
    def __init__(self, xcenter: Union[int, float], ycenter: Union[int, float], width: Union[int, float], height: Union[int, float], labelname: str, labelid: Union[str, int]) -> None:
        
        self.xcenter, self.ycenter, self.width, self.height, self.labelname, self.labelid = xcenter, ycenter, width, height, labelname, labelid
    
    def __call__(self) -> Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], str, int]:
        return self.xcenter, self.ycenter, self.width, self.height, self.labelname, self.labelid
    
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if isinstance(self.xcenter, float):
            return f"(xcenter: {self.xcenter:^.3f} | ycenter: {self.ycenter:^.3f}), (width: {self.width:^.3f} | height: {self.height:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
        return f"(xcenter: {self.xcenter: 4d} | ycenter: {self.ycenter: 4d}), (width: {self.width: 4d} | height: {self.height: 4d}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
    def ToInt(self) -> BoundingBoxCenter:
        self.xcenter, self.ycenter, self.width, self.height = tuple(map(int, [self.xcenter, self.ycenter, self.width, self.height]))
        return self

class BaseBoxes:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.Boxes: List[Union[BoundingBox, BoundingBoxCenter]] = list()
        self.LabelNameLength = 0

    def __call__(self) -> List[Union[BoundingBox, BoundingBoxCenter]]:
        return self.Boxes

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self) -> Any:
        if self.idx == len(self.Boxes):
            raise StopIteration()
        ret = self.Boxes[self.idx]
        self.idx += 1
        return ret
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        for box in self.Boxes: 
            if self.LabelNameLength < len(box.labelname): self.LabelNameLength = len(box.labelname)
        output = f"Width({self.width:^ 5}), Height({self.height:^ 5})\n"
        for idx, box in enumerate(self.Boxes):
            box = str(box).replace(box.labelname, box.labelname.center(self.LabelNameLength))
            output += f"  ({idx:^ 3}){box}\n"
        return output

    def __iadd__(self, box: Union[BoundingBox, BoundingBoxCenter]) -> BoundingBoxes:
        self.Boxes.append(box)
        return self

    def __len__(self) -> int:
        return len(self.Boxes)

    def __setitem__(self, idx: int, box: Union[BoundingBox, BoundingBoxCenter]) -> None:
        self.Boxes[idx] = box
        return None

    def __getitem__(self, idx: int) -> Union[BoundingBox, BoundingBoxCenter]:
        return self.Boxes[idx]

    def __dellitem__(self, idx: int) -> None:
        del self.Boxes[idx]
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
    def __init__(self, width: int, height: int) -> None:
        super(BoundingBoxes, self).__init__(width, height)
        self.width, self.height = width, height
        self._NormFlag = False
        self._CenterFlag = False

    def Size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def Append(self, box: Union[BoundingBox, BoundingBoxCenter]) -> BoundingBoxes:
        self.LabelNameLength = len(box.labelname) if self.LabelNameLength < len(box.labelname) else self.LabelNameLength
        self.Boxes.append(box)
        return self
    
    def ToCenter(self) -> BoundingBoxes:
        if self._CenterFlag: return self
        self._CenterFlag = True
        self._Normalize()
        for idx, box in enumerate(self.Boxes):
            if not isinstance(box, BoundingBox): continue
            xmin, ymin, xmax, ymax, labelname, labelid = box()
            box = BoundingBoxCenter((xmax + xmin) * .5, (ymax + ymin) * .5, (xmax - xmin), (ymax - ymin), labelname, labelid)
            self.__setitem__(idx, box)
        return self

    def ToPoint(self) -> BoundingBoxes:
        if not self._CenterFlag: return self
        self._CenterFlag = False
        for idx, box in enumerate(self.Boxes):
            if not isinstance(box, BoundingBoxCenter): continue
            cx, cy, w, h, labelname, labelid = box()
            (w, h) = (w * .5, h * .5)
            box = BoundingBox(cx - w, cy - h, cx + w, cy + h, labelname, labelid)
            self.__setitem__(idx, box)
        self._DNormalize()
        return self

    def _Normalize(self) -> BoundingBoxes:
        if self._NormFlag: return self
        self._NormFlag = True
        for i, b in enumerate(self.Boxes):
            A, B, C, D, L1, L2 = b()
            A /= self.width
            B /= self.height
            C /= self.width
            D /= self.height
            if isinstance(b, BoundingBox): box = BoundingBox(A, B, C, D, L1, L2)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(A, B, C, D, L1, L2)
            self.__setitem__(i, box)
        return self

    def _DNormalize(self) -> BoundingBoxes:
        if not self._NormFlag: return self
        self._NormFlag = False
        for i, b in enumerate(self.Boxes):
            A, B, C, D, L1, L2 = b()
            A = PixelRepair(A * self.width)
            B = PixelRepair(B * self.height)
            C = PixelRepair(C * self.width)
            D = PixelRepair(D * self.height)
            if isinstance(b, BoundingBox): box = BoundingBox(A, B, C, D, L1, L2)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(A, B, C, D, L1, L2)
            self.__setitem__(i, box)
        return self

    def ToInt(self) -> BoundingBoxes:
        for idx, box in enumerate(self.Boxes):
            self.__setitem__(idx, box.ToInt())
        return self
    
    def CallID(self, id) -> BoundingBoxes:
        self.ClassIDSort()
        Boxes = BoundingBoxes(self.width, self.height)
        for Box in self.Boxes:
            if Box.labelid == id:
                Boxes.Append(Box)
        return Boxes

    def ClassIDSort(self) -> BoundingBoxes:
        self.Boxes.sort(key=lambda box: box.labelid)
        return self
    
if __name__ == "__main__":

    bboxlist = BoundingBoxes(10, 10)
    bboxlist += BoundingBox(1, 2, 10, 10, "apple", 1)
    bboxlist += BoundingBox(2, 2, 10, 10, "orange", 0)
    bboxlist += BoundingBox(2, 9, 10, 10, "banana", 2)
    
    print(bboxlist.ClassIDSort())