from torch.utils.data import DataLoader, Dataset
import torch
test_datapoint_path = './test/'
train_datapoint_path = './train/'


class UCSD_train_loader(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):

        assert index < self.num_samples, 'out of the bound'
        x = torch.load(train_datapoint_path + 'train_id_' + str(index) + '.pt')
        return x

    def __len__(self):
        return self.num_samples


class UCSD_test_loader(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):

        assert index < self.num_samples, 'out of the bound'
        x = torch.load(test_datapoint_path + 'test_id_' + str(index) + '.pt')
        return x

    def __len__(self):
        return self.num_samples
