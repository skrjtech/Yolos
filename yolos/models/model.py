import torch
import torchvision

class YOLOMODEL(torch.nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20, *args, **kwargs) -> None:
        super(YOLOMODEL, self).__init__(*args, **kwargs)

        self.S, self.B, self.C = S, B, C
        self.N = B * 5 + C

class YoloLossModel(YOLOMODEL):
    def __init__(self, lambdaobj: float = 5., lambdanoobj: float = .5, *args, **kwargs):
        super(YoloLossModel, self).__init__(*args, **kwargs)

        self.lambdaobj = lambdaobj
        self.lambdanoobj = lambdanoobj

        self.CI = [4, 9]  # Confidence Index
        self.BI = [[0, 1, 2, 3], [5, 6, 7, 8]]  # BBoxIndex
        self.LI = [self.B * 5 + idx for idx in range(self.C)]  # Label Index
        self.XYI = [0, 1]  # XY or XYMin
        self.WHI = [2, 3]  # WH or XYMax

    def forward(self, P: torch.Tensor, T: torch.Tensor):
        B, C, N = self.B, self.C, self.N
        CI = self.CI
        BI = self.BI
        LI = self.LI
        XYI = self.XYI
        WHI = self.WHI

        Batch = P.size(0)

        coordMask = (T[..., 4] == 1).unsqueeze(-1).expand_as(T)
        noobjMask = (T[..., 4] == 0).unsqueeze(-1).expand_as(T)

        coordP = P[coordMask].reshape(-1, N)  # [coord_n, N]
        noobjP = P[noobjMask].reshape(-1, N)  # [coord_n, N]

        coordT = T[coordMask].reshape(-1, N)  # [coord_n, N]
        noobjT = T[noobjMask].reshape(-1, N)  # [coord_n, N]

        # Class Label
        ClassP = coordP[..., LI].reshape(-1, C)  # [coord_n, C]
        ClassT = coordT[..., LI].reshape(-1, C)  # [coord_n, C]
        # No Object Confidence
        NoObjP = noobjP[..., CI].reshape(-1, B)  # [nooobj_n, B]
        NoObjT = noobjT[..., CI].reshape(-1, B)  # [nooobj_n, B]
        # Object Confidence
        ConfP = coordP[..., CI].reshape(-1, B);  # [coord_n, B]
        # BBox
        BBoxP = coordP[..., BI].reshape(-1, B, 4)  # [coord_n, B, 4(XYXY)]
        BBoxT = coordT[..., BI].reshape(-1, B, 4)  # [coord_n, B, 4(XYXY)]

        with torch.no_grad():
            iou, iouIndex = self.IoUCul(BBoxP.reshape(-1, 4), BBoxT.reshape(-1, 4))

        Range = torch.arange(iouIndex.size(0)).long()
        BBoxP = BBoxP[Range, iouIndex].reshape(-1, 4)
        BBoxT = BBoxT[Range, iouIndex].reshape(-1, 4)
        ConfP = ConfP[Range, iouIndex]

        lossXY = torch.nn.functional.mse_loss(BBoxP[..., XYI], BBoxT[..., XYI], reduction="sum")
        lossWH = torch.nn.functional.mse_loss(torch.sqrt(BBoxP[..., WHI]), torch.sqrt(BBoxT[..., WHI]), reduction="sum")
        lossObj = torch.nn.functional.mse_loss(ConfP, iou, reduction="sum")
        lossNObj = torch.nn.functional.mse_loss(NoObjP, NoObjT, reduction="sum")
        lossClass = torch.nn.functional.mse_loss(ClassP, ClassT, reduction="sum")
        loss = (self.lambdaobj * (lossXY + lossWH) + self.lambdanoobj * (lossNObj) + (lossObj + lossClass)) / Batch
        return loss

    def IoUCul(self, P, T):
        """
        P (input): [Batch, coord_n, xywh]
        T (input): [Batch, coord_n, xywh]
        """

        XYI = self.XYI
        WHI = self.WHI

        S = 7
        P = P.clone()
        T = T.clone()

        PXYMIN = P[..., XYI] / float(S) - 0.5 * P[..., WHI]
        PXYMAX = P[..., XYI] / float(S) + 0.5 * P[..., WHI]

        TXYMIN = T[..., XYI] / float(S) - 0.5 * T[..., WHI]
        TXYMAX = T[..., XYI] / float(S) + 0.5 * T[..., WHI]

        lt = torch.max(PXYMIN, TXYMIN)
        rb = torch.min(PXYMAX, TXYMAX)

        wh = torch.clamp(rb - lt, min=0.)
        intersect = (wh[..., 0] * wh[..., 1])

        Area1 = (PXYMAX - PXYMIN)
        Area1 = Area1[..., 0] * Area1[..., 1]
        Area2 = (TXYMAX - TXYMIN)
        Area2 = Area2[..., 0] * Area2[..., 1]
        Union = Area1 + Area2 - intersect

        iou = intersect / Union
        return torch.max(iou.reshape(-1, 2), dim=1)

class YoloV1(YOLOMODEL):
    def __init__(self, *args, **kwargs) -> None:
        super(YoloV1, self).__init__(*args, **kwargs)

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
    pass