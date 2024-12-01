from ..datasets.food101 import Food101


def run():
    dataset = Food101()
    training_data = dataset.getTrainingData()
    print(type(training_data))

if __name__ == "__main__":
    run()
