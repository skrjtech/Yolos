from typing import Any

__all__ = [
    "IoUWrapper"
]

class IntersectionOverUnion:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()

class IoUWrapper(IntersectionOverUnion):
    pass

class IoU(IoUWrapper):
    def forward(self, boxA: any, boxB: any):
        pass

if __name__ == "__main__":
    pass