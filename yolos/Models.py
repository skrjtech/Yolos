import torch
import torchvision

from yolos.Utils import Norm2DNorm, IntersectionOverUnion
def yololoss(S: int, B: int, C: int, LambdaObj: float=5., LambdaNoObj: float=.5) -> torch.Tensor:
    N = B * 5 + C
    CI = [4, 9] # Confidence
    BI = [[0, 1, 2, 3], [5, 6, 7, 8]] # Boxes
    LI = [B * 5 + idx for idx in range(C)] # Labels 
    XYI = [0, 1] # XYIndex
    WHI = [2, 3] # WHIndex
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
        NoObjP = noobjP[..., CI].reshape(-1, B)  # [nooobj_n, B]
        NoObjT = noobjT[..., CI].reshape(-1, B)  # [nooobj_n, B]
        # Object Confidence
        ConfP = coordP[..., CI].reshape(-1, B);  # [coord_n, B]
        # BBox
        BBoxP = coordP[..., BI].reshape(-1, B, 4)  # [coord_n, B, 4(XYWH)]
        BBoxT = coordT[..., BI].reshape(-1, B, 4)  # [coord_n, B, 4(XYWH)]

        with torch.no_grad():
            P = BBoxP.clone()
            T = BBoxT.clone()
            P = Norm2DNorm(P, S).reshape(-1, 4)
            T = Norm2DNorm(T, S).reshape(-1, 4)
            Step = int(P.size(0) / 2 + 1)
            iou = IntersectionOverUnion(P, T[::2].reshape(-1, 4))
            iou, iouIndex = torch.max(iou.T.reshape(-1, 2)[::Step], dim=-1)

        Range = torch.arange(iouIndex.size(0)).long()
        BBoxP = BBoxP[Range, iouIndex].reshape(-1, 4)
        BBoxT = BBoxT[Range, iouIndex].reshape(-1, 4)
        ConfP = ConfP[Range, iouIndex]
        
        lossXY = torch.nn.functional.mse_loss(BBoxP[..., XYI], BBoxT[..., XYI], reduction="sum")
        lossWH = torch.nn.functional.mse_loss(torch.sqrt(BBoxP[..., WHI]), torch.sqrt(BBoxT[..., WHI]), reduction="sum")
        lossObj = torch.nn.functional.mse_loss(ConfP, iou, reduction="sum")
        lossNObj = torch.nn.functional.mse_loss(NoObjP, NoObjT, reduction="sum")
        lossClass = torch.nn.functional.mse_loss(ClassP, ClassT, reduction="sum")
        loss = (LambdaObj * (lossXY + lossWH) + LambdaNoObj * (lossNObj) + (lossObj + lossClass)) / BatchSize
        return loss
    
    return __CALL__

from yolos.YoloBoxes import YoloRoot
class YOLOMODEL(torch.nn.Module):
    pass

class YoloLossModel(YOLOMODEL, YoloRoot):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        YoloRoot.__init__(self, **kwargs)
        
    def forward(self, P: torch.Tensor, T: torch.Tensor):
        return yololoss(self.S, self.B, self.C, self.LambdaObj, self.LambdaNoObj)(P, T)

class YoloV1(YOLOMODEL, YoloRoot):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        YoloRoot.__init__(self, **kwargs)
        
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
    YoloRoot(C=3)
    BBox1 = torch.zeros(1, 7, 7, 13)
    BBox2 = torch.zeros(1, 7, 7, 13)
    BBox1[0, 1, 1] = torch.Tensor([0.11, 0.11, 0.4, 0.4, .9, 0.9, 0.9, 0.4, 0.4, .9, 0., 1., 0.])
    BBox2[0, 1, 1] = torch.Tensor([0.1, 0.1, 0.4, 0.4, 1., 0.1, 0.1, 0.4, 0.4, 1., 0., 0., 1.])
    print(YoloLossModel()(BBox1, BBox2))