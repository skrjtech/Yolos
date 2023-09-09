from typing import Any
import torchvision

class Compose:
    def __init__(self, transform: list) -> None:
        self.transform = transform

    def __call__(self, inp, bbox) -> Any:
        for trans in self.transform:
            inp, bbox = trans(inp, bbox)
        return inp, bbox


class ToTensor:
    def __init__(self, *args, **kwargs) -> None:
        self.totensor = torchvision.transforms.ToTensor(*args, **kwargs)

    def __call__(self, inp, bbox):
        return self.totensor(inp), bbox


class Resize:
    def __init__(self, *args, **kwargs) -> None:
        self.resize = torchvision.transforms.Resize(*args, **kwargs)

    def __call__(self, inp, bbox):
        return self.resize(inp), bbox