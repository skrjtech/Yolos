from __future__ import annotations
from typing import Any, Union, Tuple, List

__all__ = [
    "BaseBoxes",
    "BoundingBox",
    "BoundingBoxCenter",
    "BoundingBoxes"
]

class BoxBase:
    Values: List=None
    FormatInt: str=None
    FormatFloat: str=None
    
    _FloatFlag: bool=None
    def __call__(self) -> Any: return self.Values
    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str:
        if not self._FloatFlag: 
            return self.FormatInt.format(*self.Values)
        return self.FormatFloat.format(*self.Values)
    
    def ToInt(self) -> BoxBase:
        """
        Values: [LabelName, LabelId, ...]
        """
        self.Values[1: 5+1] = list(map(int, self.Values[1: 5+1])) # Label名以外を変換
        return self

def Str2Int(val: Any):
    if isinstance(val, str): return int(val)
    return val
    
class BoundingBox(BoxBase):
    """
    オブジェクト属性と座標を格納する.
    LabelName Labelid Xmin Ymin Xmax Ymax
    
    """
    def __init__(self, labelname: str, labelid: int, xmin: Any, ymin: Any, xmax: Any, ymax: Any) -> None:
        
        labelid, xmin, ymin, xmax, ymax = tuple(map(Str2Int, (labelid, xmin, ymin, xmax, ymax)))
        assert xmin < xmax, xmax > xmin 
        assert ymin < ymax, ymax > ymin 
        
        self._FloatFlag = isinstance(xmin, float)
        self.Values = list([labelname, int(labelid), xmin, ymin, xmax, ymax])
        self.labelname, self.labelid, self.xmin, self.ymin, self.xmax, self.ymax = self.Values
        self.FormatInt = "( objname: {} | objid: {:5d} ), ( xmin: {:4d} | ymin: {:4d} ), ( xmax: {:4d} | ymax: {:4d} )"
        self.FormatFloat = "( objname: {} | objid: {:5d} ), ( xmin: {:08.03f} | ymin: {:08.03f} ), ( xmax: {:08.03f} | ymax: {:08.03f} )"
    
    def ToInt(self) -> BoundingBox: return super().ToInt()
    

class BoundingBoxCenter(BoxBase):
    def __init__(self, labelname: str, labelid: int, xcenter: Any, ycenter: Any, width: Any, height: Any) -> None:
        
        self._FloatFlag = isinstance(xcenter, float)
        self.Values = list([labelname, int(labelid), xcenter, ycenter, width, height])
        self.labelname, self.labelid, self.xcenter, self.ycenter, self.width, self.height = self.Values
        self.FormatInt = "( objname: {} | objid: {:5} ), ( xcenter: {:4d} | ycenter: {:4d} ), ( width: {:4d} | height: {:4d} )"
        self.FormatFloat = "( objname: {} | objid: {:5} ), ( xcenter: {:08.03f} | ycenter: {:08.03f} ), ( width: {:08.03f} | height: {:08.03f} )"

    def ToInt(self) -> BoundingBoxCenter: return super().ToInt()

class BaseBoxes:
    Width: int=None
    Height: int=None
    def __init__(self) -> None:
        self.Boxes: List[Union[BoundingBox, BoundingBoxCenter]] = list()

    def __call__(self) -> List[Union[BoundingBox, BoundingBoxCenter]]: return self.Boxes
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self) -> Any:
        if self.idx == len(self.Boxes): raise StopIteration()
        self.idx += 1
        return self.Boxes[self.idx - 1]
    
    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str:
        maxStringLength = max([len(box.labelname) for box in self.Boxes]) # ボックス内のLabel名の最長文字列を選択
        output = "Width({:^ 5}), Height({:^ 5})\n".format(*self.Size())
        for idx, box in enumerate(self.Boxes):
            box = str(box).replace(box.labelname, box.labelname.center(maxStringLength))
            output += f"  ({idx:^ 3}){box}\n"
        return output

    def __iadd__(self, box: Union[BoundingBox, BoundingBoxCenter]) -> BoundingBoxes:
        self.Boxes.append(box)
        return self

    def __len__(self) -> int: return len(self.Boxes)
    def __setitem__(self, idx: int, box: Union[BoundingBox, BoundingBoxCenter]) -> None:
        self.Boxes[idx] = box
        return None
        
    def __getitem__(self, idx: int) -> Union[BoundingBox, BoundingBoxCenter]: return self.Boxes[idx]
    def __dellitem__(self, idx: int) -> None: del self.Boxes[idx]
    def Size(self) -> Tuple[int, int]: return (self.Width, self.Height)
    def ToInt(self) -> BoundingBoxes:
        for idx, box in enumerate(self.Boxes): self.__setitem__(idx, box.ToInt())
        return self
    
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
        super(BoundingBoxes, self).__init__()
        self.Width, self.Height = width, height
        self._NormFlag = False
        self._CenterFlag = False
    
    def Append(self, box: Union[BoundingBox, BoundingBoxCenter]) -> BoundingBoxes:
        self.Boxes.append(box)
        return self
    
    def ToCenter(self) -> BoundingBoxes:
        if self._CenterFlag: return self
        self._CenterFlag = True
        for idx, box in enumerate(self.Boxes):
            if not isinstance(box, BoundingBox): continue
            labelname, labelid, xmin, ymin, xmax, ymax = box()
            box = BoundingBoxCenter(labelname, labelid, (xmax + xmin) * .5, (ymax + ymin) * .5, (xmax - xmin), (ymax - ymin))
            self.__setitem__(idx, box)
        return self

    def ToPoint(self) -> BoundingBoxes:
        if not self._CenterFlag: return self
        self._CenterFlag = False
        for idx, box in enumerate(self.Boxes):
            if not isinstance(box, BoundingBoxCenter): continue
            labelname, labelid, cx, cy, w, h = box()
            (w, h) = (w * .5, h * .5)
            box = BoundingBox(labelname, labelid, cx - w, cy - h, cx + w, cy + h)
            self.__setitem__(idx, box)
        return self

    def Normalize(self) -> BoundingBoxes:
        if self._NormFlag: return self
        self._NormFlag = True
        for idx, box in enumerate(self.Boxes):
            L1, L2, A, B, C, D = box()
            A /= self.Width
            B /= self.Height
            C /= self.Width
            D /= self.Height
            if isinstance(box, BoundingBox): box = BoundingBox(L1, L2, A, B, C, D)
            elif isinstance(box, BoundingBoxCenter): box = BoundingBoxCenter(L1, L2, A, B, C, D)
            self.__setitem__(idx, box)
        return self

    def DNormalize(self) -> BoundingBoxes:
        if not self._NormFlag: return self
        self._NormFlag = False
        for i, b in enumerate(self.Boxes):
            L1, L2, A, B, C, D = b()
            A = PixelRepair(A * self.Width)
            B = PixelRepair(B * self.Height)
            C = PixelRepair(C * self.Width)
            D = PixelRepair(D * self.Height)
            if isinstance(b, BoundingBox): box = BoundingBox(L1, L2, A, B, C, D)
            elif isinstance(b, BoundingBoxCenter): box = BoundingBoxCenter(L1, L2, A, B, C, D)
            self.__setitem__(i, box)
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

    bboxlist1 = BoundingBoxes(10, 10)
    bboxlist1 += BoundingBox("apple" , 1, 1, 2, 10, 10)
    bboxlist1 += BoundingBox("orange", 0, 2, 2, 10, 10)
    bboxlist1 += BoundingBox("banana", 2, 2, 9, 10, 10)

    bboxlist2 = BoundingBoxes(100, 100)
    bboxlist2 += BoundingBox("apple" , 1, 10, 20, 100, 100)
    bboxlist2 += BoundingBox("orange", 0, 20, 20, 100, 100)
    bboxlist2 += BoundingBox("banana", 2, 20, 90, 100, 100)

    print(bboxlist1)
    print(bboxlist2)