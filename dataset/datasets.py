import pandas as pd
import os
from torch.utils.data import Dataset


class EscData(Dataset):
    def __init__(self, path_to_dataset, path_to_meta, folds=[1,2,3,4,5], transform=None, samples=None):
        path_to_dataset = os.path.expanduser(path_to_dataset)
        path_to_meta = os.path.expanduser(path_to_meta)
        self.path_to_dataset = path_to_dataset
        self.transform = transform

        meta = pd.read_csv(path_to_meta)
        path_to_data = []
        labels = []
        for filename in os.listdir(path_to_dataset):
            if int(filename.split('-')[0]) not in folds:
                continue
            path_to_data.append(os.path.join(path_to_dataset, filename))
            labels.append(int(os.path.splitext(filename)[0].split('-')[-1]))
        self.path_to_data = path_to_data
        self.labels = labels
        if samples is not None:
            self.path_to_data = self.path_to_data[:samples]
            self.labels = self.labels[:samples]

    def __len__(self):
        return len(self.path_to_data)

    def __getitem__(self, idx):
        if self.transform:
            return idx, self.labels[idx], self.transform(self.path_to_data[idx])

        return idx, self.labels[idx], self.path_to_data[idx]


if __name__ == '__main__':
    path_to_dataset = os.path.expanduser("~/data/esc/audio")
    path_to_meta = os.path.expanduser("~/data/esc/meta/esc50.csv")
    d = EscData(path_to_dataset, path_to_meta, transform=None)
    print("the number of data: %d" % len(d))
    try:
        print("the first five entries:")
        for n in range(5):
            print(d[n])
    except:
        raise IndexError("There is none or fewer than 5 data in the input path")
