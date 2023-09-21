import torch
import torchvision

def Norm2DNorm(Boxes: torch.Tensor, S: int) -> torch.Tensor:
    Target = torch.zeros_like(Boxes)
    Target[..., [0, 1]] = Boxes[..., [0, 1]] / float(S) - .5 * Boxes[..., [2, 3]]
    Target[..., [2, 3]] = Boxes[..., [0, 1]] / float(S) + .5 * Boxes[..., [2, 3]]
    return Target

def IoU(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    bbox1: Shape(N, 4)
    bbox2: Shape(M, 4)
    """

    N = bbox1.size(0)
    M = bbox2.size(0)
    
    lt = torch.max(
        bbox1[..., [0, 1]].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[..., [0, 1]].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    
    rb = torch.min(
        bbox1[..., [2, 3]].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
        bbox2[..., [2, 3]].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
    )
    
    wh = torch.clamp(rb - lt, min=0) # [wh < 0] = 0 # clip at 0
    inter = wh[..., 0] * wh[..., 1] # [N, M]

    # Compute area of the bboxes
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) # [N, ]
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) # [M, ]
    area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
    area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

    union = area1 + area2 - inter # [N, M, 2]
    iou = inter / union           # [N, M, 2]

    return iou

def yololoss(S: int, B: int, C: int, LambdaObj: float=5., LambdaNoObj: float=.5) -> torch.Tensor:
    N = B * 5 + C
    CI = [4, 9]
    BI = [0, 1, 2, 3, 5, 6, 7, 8]
    LI = [B * 5 + idx for idx in range(C)]
    XYI = [0, 1]
    WHI = [2, 3]
    def __CALL__(Prediction: torch.Tensor, Target: torch.Tensor):
        BatchSize = Prediction.size(0)
        coordMask = (Target[..., 4] == 1).unsqueeze(-1).expand_as(Target)
        noobjMask = (Target[..., 4] == 0).unsqueeze(-1).expand_as(Target)

        coordP = Prediction[coordMask].reshape(-1, N)  # [coord_n, N]
        noobjP = Prediction[noobjMask].reshape(-1, N)  # [coord_n, N]

        coordT = Target[coordMask].reshape(-1, N)  # [coord_n, N]
        noobjT = Target[noobjMask].reshape(-1, N)  # [coord_n, N]

        # Class Label
        ClassP = coordP[..., LI].reshape(-1, C)  # [coord_n, C]
        ClassT = coordT[..., LI].reshape(-1, C)  # [coord_n, C]
        # No Object Confidence
        NoObjP = noobjP[..., CI].reshape(-1, 1)  # [nooobj_n, 1]
        NoObjT = noobjT[..., CI].reshape(-1, 1)  # [nooobj_n, 1]
        # Object Confidence
        ConfP = coordP[..., CI].reshape(-1, 1);  # [coord_n, 1]
        # BBox
        BBoxP = coordP[..., BI].reshape(-1, 4)  # [coord_n, 4(XYWH)]
        BBoxT = coordT[..., BI].reshape(-1, 4)  # [coord_n, 4(XYWH)]

        with torch.no_grad():
            BBoxP = Norm2DNorm(BBoxP.reshape(-1, 4), S)
            BBoxT = Norm2DNorm(BBoxT.reshape(-1, 4), S)
            iou, iouIndex = torch.max(IoU(BBoxP, BBoxT), dim=0)

        NSize= BBoxP.size(0)
        BBoxP = BBoxP.unsqueeze(0).expand(NSize, NSize, 4)[list(range(NSize)), iouIndex]
        BBoxT = BBoxT.unsqueeze(0).expand(NSize, NSize, 4)[list(range(NSize)), iouIndex]
        ConfP = ConfP.unsqueeze(0).expand(NSize, NSize, 1)[list(range(NSize)), iouIndex]
        
        lossXY = torch.nn.functional.mse_loss(BBoxP[..., XYI], BBoxT[..., XYI], reduction="sum")
        lossWH = torch.nn.functional.mse_loss(torch.sqrt(BBoxP[..., WHI]), torch.sqrt(BBoxT[..., WHI]), reduction="sum")

        lossObj = torch.nn.functional.mse_loss(ConfP.reshape(-1), iou, reduction="sum")
        lossNObj = torch.nn.functional.mse_loss(NoObjP, NoObjT, reduction="sum")
        lossClass = torch.nn.functional.mse_loss(ClassP, ClassT, reduction="sum")
        loss = (LambdaObj * (lossXY + lossWH) + LambdaNoObj * (lossNObj) + (lossObj + lossClass)) / BatchSize
        return loss
    return __CALL__

from yolos.YoloStruct import YoloRoot

class YOLOMODEL(torch.nn.Module):
    def __init__(self, root: YoloRoot, *args, **kwargs) -> None:
        super(YOLOMODEL, self).__init__(*args, **kwargs)
        self.root = root
        self.S, self.B, self.C, self.N = self.root()

class YoloLossModel(YOLOMODEL):
    def __init__(self, root: YoloRoot, *args, **kwargs):
        super(YoloLossModel, self).__init__(root=root, *args, **kwargs)

    def forward(self, P: torch.Tensor, T: torch.Tensor):
        return yololoss(self.S, self.B, self.C, self.root.LambdaObj, self.root.LambdaNoObj)(P, T)

class YoloV1(YOLOMODEL):
    def __init__(self, root: YoloRoot, *args, **kwargs) -> None:
        super(YoloV1, self).__init__(root=root, *args, **kwargs)

        self.vgg = vgg = torchvision.models.vgg16(pretrained=True)
        vgg.features.requires_grad_()
        vgg.avgpool.requires_grad_()

        vgg.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 7 * 7, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, self.S * self.S * self.N),
            torch.nn.Sigmoid()
        )

    def forward(self, inp):
        return self.vgg(inp).reshape(-1, self.S, self.S, self.N)

if __name__ == "__main__":
    BBox1 = torch.zeros(1, 7, 7, 13)
    BBox2 = torch.zeros(1, 7, 7, 13)
    BBox1[0, 1, 1] = torch.Tensor([0.11, 0.11, 0.4, 0.4, .9, 0.9, 0.9, 0.4, 0.4, .9, 0., 1., 0.])
    BBox2[0, 1, 1] = torch.Tensor([0.1, 0.1, 0.4, 0.4, 1., 0.1, 0.1, 0.4, 0.4, 1., 0., 0., 1.])
    print(YoloLossModel(YoloRoot(C=3))(BBox1, BBox2))