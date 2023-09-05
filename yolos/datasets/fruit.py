from typing import Any
import torch
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET

import os
from glob import glob

# MyLibs
from yolos.utils import EncoderBBox, MakeTargetBBox


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


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20):
        self.S, self.B, self.C = S, B, C


class FruitsImageDataset(YoloDataset):
    def __init__(self, path: str, transform: torchvision.transforms.Compose, *args, **kwargs):
        super(FruitsImageDataset, self).__init__(*args, **kwargs)
        self.ClassName = {
            "apple": 0,
            "banana": 1,
            "orange": 2
        }

        self.images = sorted(glob(os.path.join(path, "*.jpg")))
        self.xmlano = sorted(glob(os.path.join(path, "*.xml")))
        self.num = len(self.images)

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        imagePath = self.images[index]
        xmlanPath = self.xmlano[index]

        image = Image.open(imagePath)
        image = image.convert("RGB")
        width, height = image.size

        BBoxes = []
        TreeRoot = ET.parse(xmlanPath).getroot()
        for obj in TreeRoot.findall("object"):
            BBoxes += [
                [int(obj.find("bndbox").find("xmin").text),
                 int(obj.find("bndbox").find("ymin").text),
                 int(obj.find("bndbox").find("xmax").text),
                 int(obj.find("bndbox").find("ymax").text),
                 self.ClassName[obj.find("name").text]]
            ]

        image, BBoxes = self.transform(image, BBoxes)
        EncoderBox = EncoderBBox(BBoxes, width, height, S=self.S)
        Target = MakeTargetBBox(EncoderBox, self.S, self.B, self.C)

        return image, Target


if __name__ == "__main__":
    path = "database/Fruits/train"

    transform = Compose([
        Resize(size=(224, 224)),
        ToTensor()
    ])

    dataset = FruitsImageDataset(path, transform, C=3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    inputs, targets = next(iter(dataloader))
    print(inputs.shape, targets.shape)