from model.embedding import embedded

import torch
from torch.utils.data import Dataset

class RuStoreDataset(Dataset):
    def __init__(self, data):
        self.Y_train = data['labels_str']
        self.X_train = data.drop(['labels_str'], axis=1)
        self.ord = self.categorical()

    def categorical(self):
        return {label: idx for idx, label in enumerate(sorted(self.Y_train.unique()))}

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        X = self.X_train.iloc[idx]
        Y = self.Y_train.iloc[idx]

        Y = torch.tensor(self.ord[Y])

        app_name_emb = embedded([X['app_name']])
        full_description_emb = embedded([X['full_description']])
        short_description_emb = embedded([X['shortDescription']])

        out = torch.cat((app_name_emb, full_description_emb, short_description_emb), dim=0)

        return out, Y



