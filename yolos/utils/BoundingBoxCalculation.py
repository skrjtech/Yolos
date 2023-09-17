from __future__ import annotations
from typing import Any, TypeAlias, Union

from yolos.BoundingBox import BoundingBox, BoundingBoxCenter, BoundingBoxList, BoundingBoxCenterList

Types: TypeAlias = Union[BoundingBox, BoundingBoxCenter, BoundingBoxList, BoundingBoxCenterList]
class IoU:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def IntersectionOverUnion(self, Box1: Types, Box2: Types):
        pass

if __name__ == "__main__":
    pass