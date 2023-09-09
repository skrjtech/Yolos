from typing import Union, List, Tuple

def XYFormat(val: Union[str, int, float]) -> Union[int, float]:
    if isinstance(val, str): return int(val)
    elif isinstance(val, int): return val
    elif isinstance(val, float): return val

class WHSize(object):
    def __init__(self, width: int, height: int) -> None:
        width, height = tuple(map(XYFormat, (width, height)))
        self.Width, self.Height = width, height

class XYPoint(object):
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        xmin, ymin, xmax, ymax = tuple(map(XYFormat, (xmin, ymin, xmax, ymax)))
        assert xmin < xmax, "xmin, xmax どちらかの引数が不適切"
        assert ymin < ymax, "ymin, ymax どちらかの引数が不適切"
        
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax

class BoundingBox(object):
    def __init__(self, size: WHSize) -> None:
        self.size = size
        self.Boxes: List[Tuple[XYPoint, int]]= []
    
    def append(self, point: XYPoint, label: str):
        self.Boxes.append((point, label))
    
    def __len__(self) -> int:
        return len(self.Boxes)

    def __str__(self) -> str:
        _format = "\txmin: {} | ymin: {} | xmax: {} | ymax: {} | label: {}\n"
        string = ""
        for box in self.Boxes:
            p, label = box
            string += _format.format(p.xmin, p.ymin, p.xmax, p.ymax, label)
        out = f"Size(width, height): {self.size.Width, self.size.Height}\n"
        out += f"{string}"

        return out


if __name__ == "__main__":

    bbox = BoundingBox(WHSize(100, 100))
    bbox.append(XYPoint(10, 15, 40, 55), label="apple")
    bbox.append(XYPoint(10, 15, 40, 55), label="banana")
    bbox.append(XYPoint(10, 15, 40, 55), label="orange")

    print(bbox, len(bbox))
