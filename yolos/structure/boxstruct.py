from __future__ import annotations
from dataclasses import dataclass

from typing import List

class BoxLabel:
    def __init__(self, id: int, name: str=None) -> None:
        self.id = id
        self.name = name

    def __str__(self) -> str:
        return f"{self.id} == {self.name}"

@dataclass
class Box:
    def __init__(self, xmin: any, ymin: any, xmax: any, ymax: any, label: BoxLabel) -> None:
        if isinstance(xmin, str): xmin = int(xmin)
        if isinstance(ymin, str): ymin = int(ymin)
        if isinstance(xmax, str): xmax = int(xmax)
        if isinstance(ymax, str): ymax = int(ymax)

        assert xmin < xmax, ymin < ymax

        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.label = label

    def __repr__(self) -> str:
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __str__(self) -> str:
        return f"label: [{self.label}] | xmin: {self.xmin} | ymin: {self.ymin} | xmax: {self.xmax} | ymax: {self.ymax}"
    
    @classmethod
    def __eq__(self, other: Box) -> bool:
        return self.__dict__ == other.__dict__
    
class Boxes:
    def __init__(self, width: any, height: any) -> None:
        
        if isinstance(width, str): width = int(width)
        if isinstance(height, str): height = int(height)
        
        self.width = width
        self.height = height
        self.boxes: List[Box,] = []

    def __len__(self) -> int:
        return len(self.boxes)
    
    def __getitem__(self, index) -> Box:
        self.boxes[index]
        return self
    
    def __setitem__(self, index, value) -> Boxes:
        self.boxes[index] = value
        return self

    def __iadd__(self, other: Box) -> Boxes:
        self.boxes.append(other)
        return self
    
    def __str__(self) -> str:
        
        output = f"Boxes({len(self.boxes)}): Width({self.width}) x Height({self.height})\n"
        for idx, string in enumerate(self.boxes): output += f"    ({idx})({string})\n"
        return output
    
from .yolostruct import YoloRoot
class BoxCenter:
    def __init__(self, Root: YoloRoot, )

if __name__ == "__main__":
    boxes = Boxes(222, 222)
    box1 = Box(5, 5, 10, 10, BoxLabel(0, "apple"))
    boxes += box1
    boxes += box1
    boxes += box1
    boxes += box1
    print(boxes)