import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VGG16, self).__init__(*args, **kwargs)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True)
        )

        self.labelLayer = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.Softmax(dim=-1)
        )

        self.feature = nn.ModuleList([
            self.layer1,self.layer2,self.layer3,
            self.layer4,self.layer5,
            self.fc, self.labelLayer
        ])

    def forward(self, inp):
        inp = self.layer1(inp); print(inp.shape)
        inp = self.layer2(inp); print(inp.shape)
        inp = self.layer3(inp); print(inp.shape)
        inp = self.layer4(inp); print(inp.shape)
        inp = self.layer5(inp); print(inp.shape)
        inp = self.fc(inp)
        return self.labelLayer(inp)
    
class VGG16AutoEncode(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VGG16AutoEncode, self).__init__(*args, **kwargs)
        
        vgg16 = self.vgg16 = VGG16()
        self.vgg16layer1 = vgg16.layer1 # 64 x 110 x 110
        self.vgg16layer2 = vgg16.layer2 # 128 x 53 x 53
        self.vgg16layer3 = vgg16.layer3 # 256 x 23 x 23 
        self.vgg16layer4 = vgg16.layer4 # 512 x 8 x 8
        self.vgg16layer5 = vgg16.layer5 # 512 x 1 x 1

        self.ZFuture = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=1),
            nn.ReLU(inplace=True)
        )

        self.feature = nn.ModuleList([
            vgg16.feature[:5],
            self.ZFuture,
            self.layer1, self.layer2, self.layer3, 
            self.layer4, self.layer5
        ])


    def forward(self, inp):

        inp1 = self.vgg16layer1(inp) # 64 x 110 x 110
        inp2 = self.vgg16layer2(inp1) # 128 x 53 x 53
        inp3 = self.vgg16layer3(inp2) # 256 x 23 x 23
        inp4 = self.vgg16layer4(inp3) # 512 x 8 x 8
        inp5 = self.vgg16layer5(inp4) # 512 x 1 x 1

        inp_ = self.ZFuture(inp5).reshape(-1, 512, 1, 1)

        inp_ = self.layer1(inp_) + inp4
        inp_ = self.layer2(inp_) + inp3
        inp_ = self.layer3(inp_) + inp2
        inp_ = self.layer4(inp_) + inp1
        inp = self.layer5(inp_) + inp
        return inp


class YoloNetworkVGG16(nn.Module):
    def __init__(self, S=7, B=2, C=20, *args, **kwargs) -> None:
        super(YoloNetworkVGG16, self).__init__(*args, **kwargs)

        self.S, self.B, self.C = S, B, C
        N = self.N = B * 5 + C

        vgg16 = self.vgg16 = VGG16()
        self.vgg16layer1 = vgg16.layer1
        self.vgg16layer2 = vgg16.layer2
        self.vgg16layer3 = vgg16.layer3
        self.vgg16layer4 = vgg16.layer4
        self.vgg16layer5 = vgg16.layer5

        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, S * S * N),
            nn.Sigmoid()
        )

    def forward(self, inp):
        inp = self.vgg16layer1(inp)
        inp = self.vgg16layer2(inp)
        inp = self.vgg16layer3(inp)
        inp = self.vgg16layer4(inp)
        inp = self.vgg16layer5(inp)
        return self.layer1(inp).reshape(-1, self.S, self.S, self.N)


class YoloLossModel(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambdaobj=5., lambdanoobj=.5, *args, **kwargs):
        super(YoloLossModel, self).__init__(*args, **kwargs)
        self.S, self.B, self.C = S, B, C
        self.N = B * 5 + C

        self.lambdaobj = lambdaobj
        self.lambdanoobj = lambdanoobj

        self.CS = torch.LongTensor([0, 1]) # ConfidentialSclice
        self.BS = torch.LongTensor([[2, 3, 4, 5], [6, 7, 8, 9]]) # BBoxsSlice
        self.LS = torch.LongTensor([B * 2 + i for i in range(C)]) # LabelsSlice

    def forward(self, Predict, Target):
        # print("forward ", Predict.max().item(), Target.max().item())

        B, C = self.B, self.C
        N = self.N
        CS = self.CS
        BS = self.BS
        LS = self.LS

        CoordMask = (Target[..., 0] == 1).unsqueeze(-1).expand_as(Target)
        NoobjMask = (Target[..., 0] == 0).unsqueeze(-1).expand_as(Target)
        
        CoordPred = Predict[CoordMask].reshape(-1, N)
        NoobjPred = Predict[NoobjMask].reshape(-1, N)
        
        CoordTarg = Target[CoordMask].reshape(-1, N)
        NoobjTarg = Target[NoobjMask].reshape(-1, N)
        
        # Class Label
        ClassPred = CoordPred[..., LS].reshape(-1, C)
        ClassTarg = CoordTarg[..., LS].reshape(-1, C)
        # BBox
        BBoxPred = CoordPred[..., BS].reshape(-1, B, 4)
        BBoxTarg = CoordTarg[..., BS].reshape(-1, B, 4)
        # Confidential
        ConfPred = CoordPred[..., CS].reshape(-1, B)
        ConfTarg = CoordTarg[..., CS].reshape(-1, B)
        # No Confidential
        NConfPred = NoobjPred[..., CS].reshape(-1, B)
        NConfTarg = NoobjTarg[..., CS].reshape(-1, B)

        iou, GetIndex = self.IoUCul(BBoxPred, BBoxTarg)
        # print(iou.max())
        lossXY = F.mse_loss(BBoxPred[GetIndex][..., :2], BBoxTarg[GetIndex][..., :2], reduction="sum")
        lossWH = F.mse_loss(torch.sqrt(BBoxPred[GetIndex][..., :2]), torch.sqrt(BBoxTarg[GetIndex][..., :2]), reduction="sum")
        lossObj = F.mse_loss(ConfPred[GetIndex], iou, reduction="sum")
        lossNobj = F.mse_loss(NConfPred, NConfTarg, reduction="sum")
        lossClass = F.mse_loss(ClassPred, ClassTarg, reduction="sum")

        # print(f"XY {lossXY.item():.4f} |WH {lossWH.item():.4f} |Obj {lossObj.item():.4f} |NoObj {lossNobj.item():.4f} |Class {lossClass.item():.4f}")
        
        # lossXYWH = 5. * (lossXY + lossWH)
        # lossNobj_5 = 0.5 * lossNobj
        # loss = lossXYWH + lossNobj_5 + lossObj + lossClass
        return ( (self.lambdaobj * (lossXY + lossWH)) + (self.lambdanoobj * lossNobj) + lossObj + lossClass ) / float(Predict.size(0))
        
        
    def IoUCul(self, Predict, Target):
        S = self.S
        with torch.no_grad():
            PredictXY = Predict.clone()
            PredictXY[..., :2] = PredictXY[..., :2] / float(S) - 0.5 * PredictXY[..., 2:]
            PredictXY[..., 2:] = PredictXY[..., :2] / float(S) + 0.5 * PredictXY[..., 2:]
    
            TargetXY = Target.clone()
            TargetXY[..., :2] = TargetXY[..., :2] / float(S) - 0.5 * TargetXY[..., 2:]
            TargetXY[..., 2:] = TargetXY[..., :2] / float(S) + 0.5 * TargetXY[..., 2:]
    
            lt = torch.max(PredictXY[..., :2], TargetXY[..., :2])
            rb = torch.min(PredictXY[..., 2:], TargetXY[..., 2:])
    
            wh = torch.clamp(rb - lt, min=0.)
            intersect = (wh[..., 0] * wh[..., 1])
    
            Area1 = (PredictXY[... , 2:] - PredictXY[..., :2])
            Area1 = Area1[..., 0] * Area1[..., 1]
            Area2 = (TargetXY[..., 2:] - TargetXY[..., :2])
            Area2 = Area2[..., 0] * Area2[..., 1]
            Union = Area1 + Area2 - intersect
    
            iou = intersect / Union
            iou, index = iou.max(-1)
        GetIndex = [torch.arange(iou.size(0), dtype=torch.long), index]
        return iou, GetIndex
    
if __name__ == "__main__":
    
    inp = torch.rand(1, 3, 224, 224)

    vgg16 = VGG16()
    vgg16ae = VGG16AutoEncode()
    model = YoloNetworkVGG16()

    state = vgg16ae.vgg16.feature[:5].state_dict()
    model.vgg16.feature[:5].load_state_dict(state)