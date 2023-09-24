import torch
import numpy as np

from yolos.YoloBoxes import YoloRoot
from yolos.Detects import Detect
from yolos.Models import YoloV1, YoloLossModel
from yolos.Datasets import FruitsImageDataset, Compose, Resize, ToTensor

class TrainingModel(YoloRoot):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.datasetsPath = "/yolos/database/Fruits/train"
        self.paramsSavePath = "/yolos/outputs/params.pt"

        self.net = YoloV1()
        self.net.cuda()
        self.optmizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.yololoss = YoloLossModel()

        dataset = FruitsImageDataset(
            self.datasetsPath,
            Compose([Resize(size=(224, 224)), ToTensor()])
        )

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        self.dataloadertest = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    def Run(self, epochs: int):
        for epoch in range(epochs):
            try:
                maps = []
                loss = []
                for i, (inputs, targets) in enumerate(self.dataloader):
                    self.optmizer.zero_grad()

                    inputs = torch.FloatTensor(inputs).cuda()
                    targets = torch.FloatTensor(targets).cuda()

                    output = self.net(inputs)
                    loss_ = self.yololoss(output, targets)
                    
                    loss_.backward()
                    self.optmizer.step()
                    loss.append(loss_.item())
                    with torch.no_grad():
                        maps.append(Detect(1, 1).MeanAP(output.cpu(), targets.cpu()))
                    
                print(f"epoch({epoch:04d}): ", f"loss({torch.Tensor(loss).mean().item():.4f}) | MeanAPs({torch.Tensor(maps).mean().item():.4f})")
                self.Save()

            except KeyboardInterrupt:
                break

            finally:
                self.Save()

    def OneBatchRun(self):
        
        Input, Target = next(iter(self.dataloadertest))
        Input = torch.FloatTensor(Input).cuda()
        Target = torch.FloatTensor(Target).cuda()
        
        loss = []
        while True:
            try:
                
                self.optmizer.zero_grad()
                Output = self.net(Input)
                loss_ = self.yololoss(Output, Target)
                loss_.backward()
                self.optmizer.step()
                loss.append(loss_.item())
                loss__ = torch.Tensor(loss).mean().item()
                print(f"loss({loss__:.4f})", end="\r")
                if loss[-1] < 0.01:
                    break
            
            except KeyboardInterrupt:
                self.Save()
                break
    
    def Detect(self):

        Input, Target = next(iter(self.dataloadertest))
        Input = torch.FloatTensor(Input).cuda()
        # Target = torch.FloatTensor(Target).cuda()
        Output = self.net(Input).detach().cpu()
        for i in range(4):
            P, T = Detect()(Output[i], Target[i])
            print(i, "P: ", P)
            print(i, "T: ", T)
            print(Detect().MeanAP(Output[i], Target[i]))

    def Save(self):
        self.net.cpu()
        net_state = self.net.state_dict()
        optim_state = self.optmizer.state_dict()
        torch.save(
            {
                "net": net_state,
                "optim": optim_state
            },
            self.paramsSavePath
        )
        self.net.cuda()

    def Load(self):
        self.net.cpu()
        params = torch.load(self.paramsSavePath)
        self.net.load_state_dict(params["net"])
        self.net.cuda()

        self.optmizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.optmizer.load_state_dict(params["optim"])


def main():
    YoloRoot(C=3)
    Train = TrainingModel()
    Train.OneBatchRun()

def Test():
    YoloRoot(C=3)
    Train = TrainingModel()
    # Train.Load()
    Train.Detect()


if __name__ == "__main__":
    main()