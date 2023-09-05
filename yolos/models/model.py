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

    def Testing(self, predict: torch.Tensor):
        S, N = self.S, self.N
        CI = self.CI
        BI = self.BI
        LI = self.LI

        Target = torch.zeros((1, S, S, N))
        # X = 1, Y = 1
        Target[:, 1, 1, CI] = torch.Tensor([1., 1.])
        Target[:, 1, 1, BI] = torch.Tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
        Target[:, 1, 1, LI] = torch.Tensor([1., 0., 0.])

        # X = 4, Y = 4
        Target[:, 4, 4, CI] = torch.Tensor([1., 1.])
        Target[:, 4, 4, BI] = torch.Tensor([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]])
        Target[:, 4, 4, LI] = torch.Tensor([0., 1., 0.])

        # X = 6, Y = 6
        Target[:, 6, 6, CI] = torch.Tensor([1., 1.])
        Target[:, 6, 6, BI] = torch.Tensor([[0.7, 0.7, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])
        Target[:, 6, 6, LI] = torch.Tensor([0., 0., 1.])

        return self.forward(predict, Target)


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

    def Testing(self):
        with torch.no_grad():
            return self.forward(torch.rand(1, 3, 224, 224))


def makeBBoxes():
    BBoxes = []
    while len(BBoxes) < 3:
        x1 = torch.randint(0, 350, size=(1,))
        y1 = torch.randint(0, 350, size=(1,))
        x2 = torch.randint(50, 400, size=(1,))
        y2 = torch.randint(50, 400, size=(1,))
        label = torch.randint(0, 3, size=(1,))
        if x1 < x2 and y1 < y2:
            BBoxes += [[x1, y1, x2, y2, label]]
    return torch.Tensor(BBoxes)


class TrainningTest(YOLOMODEL):
    def __init__(self, *args, **kwargs) -> None:
        super(TrainningTest, self).__init__(*args, **kwargs)
        self.net = YoloV1(C=3)
        self.net.cuda()
        self.lossModel = YoloLossModel(C=3)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999))

        from utils import EncoderBBox, MakeTargetBBox
        self.BBoxes = makeBBoxes()
        self.enc = EncoderBBox(self.BBoxes, 400, 400)
        self.Target = MakeTargetBBox(self.enc, 7, 2, 3).unsqueeze(0)

    def forward(self):
        Data = torch.zeros(*(1, 3, 224, 224))
        Data = torch.FloatTensor(Data).cuda()

        S, N = self.S, self.N
        CI = self.lossModel.CI
        BI = self.lossModel.BI
        LI = self.lossModel.LI

        Target = torch.FloatTensor(self.Target).cuda()

        loss = 0.
        while not ((loss > 0.) and (loss < 0.001)):
            self.optim.zero_grad()
            pred = self.net(Data)
            loss = self.lossModel(pred, Target)
            loss.backward()
            self.optim.step()
        print(f"loss: {loss.item(): .5f}")
        pred = pred.detach().cpu().squeeze()
        return pred


if __name__ == "__main__":
    # net = YoloV1(C=3)
    # lossModel = YoloLossModel(C=3)
    # predict = net.Testing()
    # loss = lossModel.Testing(predict)
    # print(loss)
    Trainer = TrainningTest(C=3)
    Trainer()