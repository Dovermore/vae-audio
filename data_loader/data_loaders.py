from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset import transformers, CollectData, NsynthSubset


class CollectDataLoader(BaseDataLoader):
    """
    Spectrogram data loader; inheritate BaseDataLoader which inheritates PyTorch DataLoader,
    refer to DataLoader in PyTorch Doc. for further details.
    Additional transformations applied to spectrograms include:
        1. Load spectrograms that were preprocessed and stored in data_dir
        2. Perform SpecChunking to slice spectrograms into fixed-length, non-overlapping chunks
    TODO:
        [] Prolly make self.transform as arguments in config file
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, **kwargs):
        self.transform = transforms.Compose([
            transformers.LoadNumpyAry(),
            transformers.SpecChunking(duration=0.5, sr=22050, hop_size=735, reverse=False)
        ])

        self.data_dir = data_dir
        self.dataset = CollectData(self.data_dir, transform=self.transform, **kwargs)
        super(CollectDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NsynthSubsetLoader(BaseDataLoader):
    """
    The DataLoader is intended to load the *pre-computed spectrograms* stored in data_dir.
    Additional transformations applied to spectrograms include:
        1. Load spectrograms that were preprocessed and stored in data_dir
        2. Normalization (optional)
    TODO:
        [] Prolly make self.transform as arguments in config file
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, **kwargs):
        self.transform = transforms.Compose([
            transformers.LoadNumpyAry(),
        ])

        self.data_dir = data_dir
        self.dataset = NsynthSubset(self.data_dir, file_ext='.npy', transform=self.transform, **kwargs)
        super(NsynthSubsetLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    pass