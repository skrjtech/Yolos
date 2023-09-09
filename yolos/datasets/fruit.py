from typing import Any
import torch
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET

import os
from glob import glob

# MyLibs
from yolos.structure import YoloRoot, YoloBox, YoloVersion
from yolos.structure.boxstruct import Boxes, BoxLabel
from yolos.utils import EncoderBBox, MakeTargetBBox

class FruitLabel(object):

    label: dict = {

        "apple":  0,
        "banana": 1,
        "orange": 2,
        
        0: "apple",
        1: "banana",
        2: "orange"       
    }

    def __getitem__(self, key):
        return self.label[key]

class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20):
        self.S, self.B, self.C = S, B, C


class FruitsImageDataset(YoloDataset):
    def __init__(self, path: str, transform: torchvision.transforms.Compose, test: bool=False, *args, **kwargs):
        super(FruitsImageDataset, self).__init__(*args, **kwargs)
        self.ClassName = {
            "apple": 0,
            "banana": 1,
            "orange": 2
        }
        self.test = test
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

        return self._Output(BBoxes, image, Target, width, height)
    
    def _Output(self, BBoxes, images, Target, Width, Height):
        if self.test:
            return BBoxes, images, Target, (Width, Height)
        
        return images, Target

class FruitsImageDatasetTest(FruitsImageDataset):
    def __init__(self, path: str, transform: torchvision.transforms.Compose, *args, **kwargs):
        super(FruitsImageDatasetTest, self).__init__(path, transform, test=True, *args, **kwargs)


from .Base import DataSetBase
from yolos.structure import YoloRoot
from yolos.structure import Box, Boxes, BoxLabel
class FruitsImageDataset(DataSetBase):
    classNameWithIndex: dict = {
        "apple": 0, "banana": 1, "orange": 2,
        0: "apple", 1: "banana", 2: "orange"
    }
    def __init__(self, Root: YoloRoot, *args, **kwargs):
        super().__init__(Root, *args, **kwargs)

        self.images = sorted(glob(os.path.join(self.path, "*.jpg")))
        self.xmlano = sorted(glob(os.path.join(self.path, "*.xml")))
        self.num = len(self.images)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        imagePath = self.images[index]
        xmlanPath = self.xmlano[index]

        image = Image.open(imagePath)
        image = image.convert("RGB")
        width, height = image.size

        BBoxes = Boxes(width, height)
        TreeRoot = ET.parse(xmlanPath).getroot()
        for obj in TreeRoot.findall("object"):
            xmin = obj.find("bndbox").find("xmin").text
            ymin = obj.find("bndbox").find("ymin").text
            xmax = obj.find("bndbox").find("xmax").text
            ymax = obj.find("bndbox").find("ymax").text
            label = obj.find("name").text
            BBoxes += Box(xmin, ymin, xmax, ymax, BoxLabel(self.classNameWithIndex[label], label))

        image, BBoxes = self.transform((image, BBoxes))
        EncoderBox = EncoderBBox(BBoxes, width, height, S=self.S)
        Target = MakeTargetBBox(EncoderBox, self.S, self.B, self.C)