import torch
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET

import os
import glob
from typing import Any, Tuple

from yolos.BoundingBox import BoundingBoxes, BoundingBox
from yolos.YoloStruct import YoloRoot, YoloStruct

class Compose:
    def __init__(self, transform: list) -> None:
        self.transform = transform

    def __call__(self, inp, bbox) -> Tuple[torch.Tensor, BoundingBoxes]:
        for trans in self.transform:
            inp, bbox = trans(inp, bbox)
        return inp, bbox

class ToTensor:
    def __init__(self, *args, **kwargs) -> None:
        self.totensor = torchvision.transforms.ToTensor(*args, **kwargs)

    def __call__(self, inp, bbox) -> Tuple[torch.Tensor, BoundingBoxes]:
        return self.totensor(inp), bbox

class Resize:
    def __init__(self, *args, **kwargs) -> None:
        self.resize = torchvision.transforms.Resize(*args, **kwargs)

    def __call__(self, inp, bbox) -> Tuple[torch.Tensor, BoundingBoxes]:
        return self.resize(inp), bbox

def transform_(x, y) -> Tuple[torch.Tensor, BoundingBoxes]:
    return (x, y)

class FruitsImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: YoloRoot, path: str, transform: torchvision.transforms.Compose=transform_, test: bool=False, *args, **kwargs):
        super(FruitsImageDataset, self).__init__(*args, **kwargs)
        self.root = root

        self.ClassName = {
            "apple": 0,
            "banana": 1,
            "orange": 2
        }

        self.test = test
        self.images = sorted(glob.glob(os.path.join(path, "*.jpg")))
        self.xmlano = sorted(glob.glob(os.path.join(path, "*.xml")))
        self.num = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        imagePath = self.images[index]
        xmlanPath = self.xmlano[index]

        image = Image.open(imagePath)
        image = image.convert("RGB")
        width, height = image.size

        BBoxes = BoundingBoxes(width, height)
        TreeRoot = ET.parse(xmlanPath).getroot()
        for obj in TreeRoot.findall("object"):
            BBoxes += BoundingBox(
                obj.find("bndbox").find("xmin").text,
                obj.find("bndbox").find("ymin").text,
                obj.find("bndbox").find("xmax").text,
                obj.find("bndbox").find("ymax").text,
                obj.find("name").text,
                self.ClassName[obj.find("name").text]
            )

        image, BBoxes = self.transform(image, BBoxes)
        yoloStruct = YoloStruct(self.root, BBoxes.Normalize().ToCenter())
        return image, yoloStruct


class FruitsImageDatasetTest(FruitsImageDataset):
    def __init__(self, path: str, transform: torchvision.transforms.Compose, *args, **kwargs):
        super(FruitsImageDatasetTest, self).__init__(path, transform, test=True, *args, **kwargs)

def collate_fn(batch):
    images, targets= list(zip(*batch))
    images = torch.stack(images)
    targets = targets
    return images, targets

if __name__ == "__main__":
    Datasets = FruitsImageDataset(YoloRoot(C=3), "database/Fruits/train", transform=Compose([Resize(size=(224, 224)), ToTensor()]))
    Dataloader = torch.utils.data.DataLoader(Datasets, batch_size=4, shuffle=True, collate_fn=collate_fn)
    images, Target = Datasets[0]
    # print(Target.Encoder().ToTarget().shape)
    # print(images.shape)
    img, tar = next(iter(Dataloader))
    print(tar)