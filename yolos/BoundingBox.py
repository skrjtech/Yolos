from __future__ import annotations
from ast import Tuple
from typing import Any, Union, List

__all__ = [
    "BoundingBox",
    "BoundingBoxCenter",
    "BoundingBoxList",
    "BoundingBoxCenterList",
    "Center2XYPointer"
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
        return f"(xmin: {self.xmin:^ 5} | ymin: {self.ymin:^ 5}), (xmax: {self.xmax:^ 5} | ymax: {self.ymax:^ 5}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
    def Int(self):
        self.xmin, self.ymin, self.xmax, self.ymax = tuple(map(int, [self.xmin, self.ymin, self.xmax, self.ymax])) 
    
class BoundingBoxCenter:
    def __init__(self, xcenter: Union[int, float], ycenter: Union[int, float], width: Union[int, float], height: Union[int, float], labelname: str, labelid: Union[str, int]) -> None:
        self.xcenter, self.ycenter, self.width, self.height, self.labelname, self.labelid = xcenter, ycenter, width, height, labelname, labelid
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.xcenter, self.ycenter, self.width, self.height, self.labelname, self.labelid

    def __str__(self) -> str:
        return f"(xcenter: {self.xcenter:^ 5} | ycenter: {self.ycenter:^ 5}), (width: {self.width:^ 5} | height: {self.height:^ 5}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
    def Int(self):
        self.xcenter, self.ycenter, self.width, self.height = tuple(map(int, [self.xcenter, self.ycenter, self.width, self.height])) 

class _BaseBoxList:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.Box: List[Union[BoundingBox, BoundingBoxCenter]] = list()

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self) -> Any:
        if self.idx == len(self.Box):
            raise StopIteration()
        ret = self.Box[self.idx]
        self.idx += 1
        return ret
    
    def Size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def append(self, bbox: Union[BoundingBox, BoundingBoxCenter]) -> Any:
        self.Box += [bbox]
        return None
    
    def __str__(self) -> str:
        output = ""
        for idx, b in enumerate(self.Box):
            output += f"  ({idx:^ 3}){b}\n"
        return output

    def __iadd__(self, bbox: Union[BoundingBox, BoundingBoxCenter]) -> Any:
        self.Box += [bbox]
        return self

    def __len__(self) -> int:
        return len(self.Box)

    def __setitem__(self, idx: int, bbox: Union[BoundingBox, BoundingBoxCenter]) -> None:
        self.Box[idx] = bbox
        return None

    def __getitem__(self, idx: int) -> Any:
        return self.Box[idx]

    def __dellitem__(self, idx: int) -> None:
        del self.Box[idx]
        return None
    
    def Normalize(self):
        for i, b in enumerate(self.Box):
            A, B, C, D, L1, L2 = b()
            A /= self.width
            B /= self.height
            C /= self.width
            D /= self.height
            if isinstance(b, BoundingBox): box = BoundingBox(A, B, C, D, L1, L2)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(A, B, C, D, L1, L2)
            self.__setitem__(i, box)

    def DNormalize(self):
        for i, b in enumerate(self.Box):
            A, B, C, D, L1, L2 = b()
            A *= self.width
            B *= self.height
            C *= self.width
            D *= self.height
            if isinstance(b, BoundingBox): box = BoundingBox(A, B, C, D, L1, L2)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(A, B, C, D, L1, L2)
            self.__setitem__(i, box)

    def Int(self) -> Union[BoundingBoxList, BoundingBoxCenterList]:
        for b in self.Box: b.Int()
        return self

class BoundingBoxList(_BaseBoxList):
    """
    BBoxオブジェクトの格納する.
    """

    #BoxList: List[BoundingBox] = list()
    def __init__(self, width: int, height: int) -> None:
        super(BoundingBoxList, self).__init__(width, height)
        self.width, self.height = width, height
    
    def __str__(self) -> str:
        output = f"Width({self.width:^ 5}), Height({self.height:^ 5})\n"
        return output + super().__str__()
    
    def append(self, bbox: BoundingBox) -> None:
        return super().append(bbox)
    
    def __iadd__(self, bbox: BoundingBox) -> BoundingBoxList:
        return super().__iadd__(bbox)

    def __len__(self) -> int:
        return super().__len__()

    def __setitem__(self, idx:int, bbox: BoundingBox) -> None:
        return super().__setitem__(idx, bbox)

    def __getitem__(self, idx: int) -> BoundingBox:
        return super().__getitem__(idx)

class BoundingBoxCenterList(_BaseBoxList):
    """
    (Xmin,Ymin,Xmax,Ymin) To (XCenter,YCenter,Width, Height)\n
    BoundingBoxListを受け取ると中心座標に変換される.
    """

    def __init__(self, bboxlist: BoundingBoxList) -> None:
        super(BoundingBoxCenterList, self).__init__(*bboxlist.Size())
        self._xyxy2center(bboxlist)

    def _xyxy2center(self, bboxlist: BoundingBoxList):
        for bbox in bboxlist:
            xmin, ymin, xmax, ymax, labelname, labelid = bbox()
            center = BoundingBoxCenter((xmax + xmin) * .5, (ymax + ymin) * .5, (xmax - xmin), (ymax - ymin), labelname, labelid)
            self.append(center)
    
    def __str__(self) -> str:
        output = f"Width({self.width:^ 5}), Height({self.height:^ 5})\n"
        return output + super().__str__()

def Center2XYPointer(bboxcenter: BoundingBoxCenterList) -> BoundingBoxList:
    _BCenter = BoundingBoxList(*bboxcenter.Size())
    def _(boxcenter: BoundingBoxCenter):
        xc, yc, w, h, labelname, labelid = boxcenter()
        return BoundingBox(
            xc - .5 * w,
            yc - .5 * h,
            xc + .5 * w,
            yc + .5 * h,
            labelname, labelid
        )
    
    for bc in bboxcenter:
        _BCenter.append(_(bc))
    return _BCenter 

if __name__ == "__main__":
    bboxlist = BoundingBoxList(10, 10)
    bboxlist += BoundingBox(1, 2, 10, 10, "apple", 1)
    bboxlist += BoundingBox(2, 2, 10, 10, "orange", 0)
    bboxlist += BoundingBox(2, 9, 10, 10, "banana", 2)
    print(bboxlist)

    bboxlist.Normalize()
    print(bboxlist)

    # bboxlist.DNormalize()
    # print(bboxlist)

    BCenter = BoundingBoxCenterList(bboxlist)
    print(BCenter)
    
    BCenter.DNormalize()
    print(BCenter)

    Pointer = Center2XYPointer(BCenter)
    print(Pointer)