# import torch
# from yolos.structure import YoloRoot, Box, Boxes

# __all__ = [
#     "DatasetBase"
# ]

# class DataSetBase(torch.utils.data.Dataset):
#     def __init__(self, Root: YoloRoot, path: str, transform: any=lambda x: x):
#         self.root = Root
#         self.yolobox = Root.yolobox

#         self.path = path
#         self.transform = transform