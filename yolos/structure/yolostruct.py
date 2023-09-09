from enum import Enum
from typing import Union, Tuple, List

__all__ = [
    "YoloVersion",
    "YoloBox",
    "YoloRoot",
]

class YoloVersion(Enum):
    YOLOV1 = 1
    YOLOV2 = 2
    YOLOV3 = 3
    YOLOV4 = 4
    YOLOV5 = 5
    YOLOV6 = 6
    YOLOV7 = 7
    YOLOV8 = 8
    MYYOLO = 0

class YoloBox:
    def __init__(self, S: int, B: int, C: int) -> None:
        self.S, self.B, self.C = S, B, C
        self.N = B * 5 + C

    def __eq__(self, other) -> bool:
        S = self.S == other.S
        B = self.B == other.B
        C = self.C == other.C
        return S and B and C

yoloboxtype = Union[Tuple, List, YoloBox]
yoloversiontype = Union[int, YoloVersion]
othertypes = Union[int, yoloboxtype, yoloversiontype]
class YoloRoot:
    def __init__(self, yolobox: yoloboxtype, yoloversion: yoloversiontype) -> None:
        
        if isinstance(yolobox, Tuple) or isinstance(yolobox, List): yolobox = YoloBox(*yolobox)
        if isinstance(yoloversion, int): yoloversion = YoloVersion(yoloversion)
        self.yolobox = yolobox
        self.yoloversion = yoloversion

    def __eq__(self, other: othertypes) -> bool:
        if isinstance(other, YoloVersion):
            return other == self.yoloversion

        if isinstance(other, YoloBox):
            return other == self.yolobox


if __name__ == "__main__":
    version1 = Enum("YoloVersion", ["YOLOV1", "YOLOV2", "YOLOV3", "YOLOV4", "YOLOV5", "YOLOV6", "YOLOV7", "YOLOV8", "MYYOLO"])
    version2 = YoloVersion(2)
    print(version2)

    root = YoloRoot((7, 2, 3), 1)
    print(root == YoloVersion(0))
    print(root == YoloBox(7, 2, 3))
