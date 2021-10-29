from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class NoiseDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform
        import_data = 'cassens_test'
        raw = pd.read_csv(
            './input/inputs_' + import_data + '.csv',
            dtype='str',
            sep=';',
            header=0,
            index_col=[0, 1],
            float_precision='round_trip',
            na_values="none"
        )

        for key in raw.columns:
            mask = raw[key].isna()
            if sum(mask) > 0:
                print(f"{key} : {sum(mask)}")
        raw.fillna(0, inplace=True)
        raw = raw.astype('float32')
        y = raw.iloc[:,-2:]
        #y = raw['kij']
        X = raw.iloc[:, 2:-2].to_numpy(dtype=np.float32)
        X = StandardScaler().fit_transform(X)
        self.x = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y.to_numpy(dtype=np.float32)).reshape(-1, 2).to(device)
        self.n_samples = X.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        self.size = len(self.x)
        return self.size


def main():
    import matplotlib.pyplot as plt

    dataset = NoiseDataset()
    plt.scatter(dataset.x, dataset.y, marker=".")
    plt.show()


if __name__ == "__main__":
    main()
