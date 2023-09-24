# Standard Package
from typing import Tuple
# FrameWork
import torch

def Norm2DNorm(Boxes: torch.Tensor, S: int) -> torch.Tensor:
    Target = torch.zeros_like(Boxes)
    Target[..., [0, 1]] = Boxes[..., [0, 1]] / float(S) - .5 * Boxes[..., [2, 3]]
    Target[..., [2, 3]] = Boxes[..., [0, 1]] / float(S) + .5 * Boxes[..., [2, 3]]
    return Target

def IntersectionOverUnion(BBoxP: torch.Tensor, BBoxT: torch.Tensor):

    N, M = BBoxP.size(0), BBoxT.size(0)
    PXYMIN = BBoxP[..., [0, 1]].unsqueeze(1).expand(N, M, 2)
    PXYMAX = BBoxP[..., [2, 3]].unsqueeze(1).expand(N, M, 2)
    TXYMIN = BBoxT[..., [0, 1]].unsqueeze(0).expand(N, M, 2)
    TXYMAX = BBoxT[..., [2, 3]].unsqueeze(0).expand(N, M, 2)

    Min, Max = torch.max(PXYMIN, TXYMIN), torch.min(PXYMAX, TXYMAX)
    
    WH = torch.clamp(Max - Min, min=0.)
    Intersection = (WH[..., 0] * WH[..., 1])
    Area1 = (PXYMAX - PXYMIN)
    Area1 = Area1[..., 0] * Area1[..., 1]
    Area2 = (TXYMAX - TXYMIN)
    Area2 = Area2[..., 0] * Area2[..., 1]
    Union = Area1 + Area2 - Intersection
    iou = Intersection / Union
    return iou

def NMS(Boxes: torch.Tensor, Scores: torch.Tensor, threshold: float=0.5, top_k: int=200) -> Tuple[torch.Tensor, int]:
        count = 0
        keep = Scores.new(Scores.size(0)).zero_().long()
        x1 = Boxes[:, 0]
        y1 = Boxes[:, 1]
        x2 = Boxes[:, 2]
        y2 = Boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        tmp_x1 = Boxes.new()
        tmp_y1 = Boxes.new()
        tmp_x2 = Boxes.new()
        tmp_y2 = Boxes.new()
        tmp_w =  Boxes.new()
        tmp_h =  Boxes.new()
        v, idx = Scores.sort(0)
        idx = idx[-top_k:]
        while idx.numel() > 0:
            i = idx[-1]
            
            keep[count] = i
            count += 1
            
            if idx.size(0) == 1: break
            idx = idx[:-1]
            
            tmp_x1 = torch.index_select(x1, 0, idx)
            tmp_y1 = torch.index_select(y1, 0, idx)
            tmp_x2 = torch.index_select(x2, 0, idx)
            tmp_y2 = torch.index_select(y2, 0, idx)
            
            tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
            tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
            tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
            tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

            tmp_w.resize_as_(tmp_x2)
            tmp_h.resize_as_(tmp_y2)

            tmp_w = tmp_x2 - tmp_x1
            tmp_h = tmp_y2 - tmp_y1

            tmp_w = torch.clamp(tmp_w, min=0.0)
            tmp_h = torch.clamp(tmp_h, min=0.0)

            inter = tmp_w*tmp_h

            rem_areas = torch.index_select(area, 0, idx)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union

            idx = idx[IoU.le(threshold)]
            
        return keep, count

def AP(Scores: torch.Tensor, Correct: torch.Tensor) -> torch.Tensor:
        
        if torch.sum(Correct) == 0: return torch.Tensor([0.])

        IndexSort = torch.sort(Scores, descending=True)[-1] # 降順
        Correct = Correct[IndexSort]

        TP = torch.cumsum(Correct, dim=-1)
        Precision = TP / (torch.arange(TP.size(0)) + 1.)
        Recall = TP / torch.sum(Correct, dim=-1)
        
        # PrecisionFlip = Precision.flip(dims=(0,))
        # PrecisionFlip = torch.cummax(PrecisionFlip, dim=0)[0].flip(dims=(0,))

        Precision = torch.concat([torch.Tensor([0]), Precision, torch.Tensor([0])], dim=-1)
        Recall = torch.concat([torch.Tensor([0]), Recall, torch.Tensor([1])], dim=-1)

        Recall = Recall[1:] - Recall[:-1]
        
        return torch.sum(Recall * Precision[1:], dim=-1)

def CreateBoxes(numBoxes: int, ClassNum: int=3, width: int=100, height: int=100):
    Boxes = []
    for _ in range(numBoxes):  
        id = torch.randint(ClassNum, size=(1,))
        while True:
            xmin = torch.randint(width, size=(1,))
            ymin = torch.randint(height, size=(1,))
            xmax = torch.randint(width, size=(1,))
            ymax = torch.randint(height, size=(1,))
            Box = torch.Tensor([id, xmin, ymin, xmax, ymax])
            if xmin < xmax and ymin < ymax: break
        Boxes.append(Box)
    return torch.stack(Boxes)

if __name__ == "__main__":
    # print(CreateBoxes(5))
    torch.manual_seed(123)
    P = CreateBoxes(3)
    T = CreateBoxes(2)
    print(P)
    print(T)
    iou = IntersectionOverUnion(T, P)
    print(iou)