import torch

# MyLibs
import utils as myutils
import models as mymodels
import Datasets as mydatasets

class TrainingModel:
    def __init__(self) -> None:

        self.net = mymodels.YoloV1(C=3)
        self.net.cuda()
        self.optmizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.yololoss = mymodels.YoloLossModel(C=3)

        dataset = mydatasets.FruitsImageDataset(
            "/yolosProject/yolos/Database/Fruits/train",
            mydatasets.Compose([mydatasets.Resize(size=(224, 224)), mydatasets.ToTensor()]),
            C=3
        )
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    def Run(self, epochs: int):

        for epoch in range(epochs):
            try:
                loss = 0
                for i, (inputs, targets) in enumerate(self.dataloader):
                    self.optmizer.zero_grad()
                    inputs = torch.FloatTensor(inputs).cuda()
                    targets = torch.FloatTensor(targets).cuda()

                    output = self.net(inputs)
                    loss_ = self.yololoss(output, targets)
                    loss_.backward()
                    self.optmizer.step()
                    loss += loss_.item()
                print(f"epoch({epoch: 04}): ", f"{loss / len(self.dataloader): .4f}")
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
            "params.pt"
        )
        self.net.cuda()

    def Load(self):
        self.net.cpu()
        params = torch.load("params.pt")
        self.net.load_state_dict(params["net"])
        self.net.cuda()

        self.optmizer = torch.optim.adam(self.net.parameters(), lr=0.0001)
        self.optmizer.load_state_dict(params["optim"])


def main():
    Train = TrainingModel()
    Train.Run(10)


if __name__ == "__main__":
    main()