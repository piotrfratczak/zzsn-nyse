from core.train import train, test
from utils.setup import parse_args, get_model
from features.make_dataset import DatasetMaker


def main() -> None:
    args = parse_args()
    dm = DatasetMaker()
    data_loaders = dm.make_dataset(args)
    model = get_model(args)
    train(model, data_loaders, args)


if __name__ == '__main__':
    main()
