from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset import transformers, EscData


class EscDataLoader(BaseDataLoader):
    """
    Spectrogram data loader; inheritate BaseDataLoader which inheritates PyTorch DataLoader,
    refer to DataLoader in PyTorch Doc. for further details.
    Additional transformations applied to spectrograms include:
        1. Load spectrograms that were preprocessed and stored in data_dir
        2. Perform SpecChunking to slice spectrograms into fixed-length, non-overlapping chunks
    TODO:
        [] Prolly make self.transform as arguments in config file
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, duration=2.5, **kwargs):
        self.transform = transforms.Compose([
            transformers.LoadNumpyAry(),
            transformers.SpecChunking(duration=duration, sr=22050, hop_size=735, reverse=False)
        ])

        self.data_dir = data_dir
        self.dataset = EscData(self.data_dir, transform=self.transform, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CatEscDataLoader(BaseDataLoader):
    """
    Spectrogram data loader; inheritate BaseDataLoader which inheritates PyTorch DataLoader,
    refer to DataLoader in PyTorch Doc. for further details.
    Additional transformations applied to spectrograms include:
        1. Load spectrograms that were preprocessed and stored in data_dir
        2. Perform SpecChunking to slice spectrograms into fixed-length, non-overlapping chunks
    TODO:
        [] Prolly make self.transform as arguments in config file
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, duration=2.5, **kwargs):
        self.transform = transforms.Compose([
            transformers.LoadNumpyAry(),
            transformers.SpecChunking(duration=duration, sr=22050, hop_size=735, reverse=False)
        ])

        self.data_dir = data_dir
        self.dataset = EscData(self.data_dir, transform=self.transform, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    pass