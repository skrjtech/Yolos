import torch
import numpy as np

from yolos.YoloStruct import YoloRoot, Detect
from yolos.Models import YoloV1, YoloLossModel
from yolos.Datasets import FruitsImageDataset, Compose, Resize, ToTensor, collate_fn

class TrainingModel:
    def __init__(self) -> None:

        self.datasetsPath = "database/Fruits/train"
        self.paramsSavePath = "outputs/params.pt"

        self.root = YoloRoot(C=3)

        self.net = YoloV1(self.root)
        self.net.cuda()
        self.optmizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.yololoss = YoloLossModel(self.root)

        dataset = FruitsImageDataset(
            self.root,
            self.datasetsPath,
            Compose([Resize(size=(224, 224)), ToTensor()])
        )
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    def Run(self, epochs: int):
        for epoch in range(epochs):
            try:
                MAPs = []
                loss = 0.
                for i, (inputs, targets) in enumerate(self.dataloader):
                    self.optmizer.zero_grad()

                    inputs = torch.FloatTensor(inputs).cuda()
                    tar = [box.Encoder().ToTarget() for box in targets]
                    tar = torch.stack(tar)
                    T = torch.FloatTensor(tar).cuda()
                    output = self.net(inputs)
                    loss_ = self.yololoss(output, T)
                    loss_.backward()
                    self.optmizer.step()
                    loss += loss_.item()
                    MAP = []
                    output = output.detach().cpu()
                    for idx, T in enumerate(targets):
                        detect = Detect(T)
                        MAP.append(detect.MeanAP(output[idx]))
                    MAP = torch.mean(torch.Tensor(MAP)).item()
                    MAPs.append(np.nan_to_num(MAP))
                print(f"epoch({epoch:04d}): ", f"{loss / len(self.dataloader):.4f}", f"MAP: {np.mean(MAPs):.4f}")
                self.Save()

            except KeyboardInterrupt:
                break

            finally:
                self.Save()

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

        self.optmizer = torch.optim.adam(self.net.parameters(), lr=0.0001)
        self.optmizer.load_state_dict(params["optim"])


def main():
    Train = TrainingModel()
    Train.Run(1000)

if __name__ == "__main__":
    main()