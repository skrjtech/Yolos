from yolos.training import TrainingModel

def main():
    model = TrainingModel()
    model.paramsSavePath = "outputs/params.pt"
    model.Run(1000)

if __name__ == '__main__':
    main()