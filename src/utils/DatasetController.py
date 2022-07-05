import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
import torchvision

# custom packages
from ..config import config

logger = logging.getLogger(__name__)


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, batch_size=config.BATCH_SIZE, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

    def __add__(self, other):
        assert self.tensors[0][0].size(0) == other.tensors[0][0].size(0)
        data = torch.cat((self.tensors[0], other.tensors[0]), 0)
        label = torch.cat((self.tensors[1], other.tensors[1]), 0)

        self.tensors = (data, label)

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True)

    def class_swap(self, class_1, class_2):
        labels = self.tensors[1].numpy()
        class_1_labels = np.array(labels == class_1).astype(int)
        class_2_labels = np.array(labels == class_2).astype(int)

        class_1_labels = class_1_labels * (class_2 - class_1)
        class_2_labels = class_2_labels * (class_1 - class_2)

        labels += class_1_labels + class_2_labels
        self.tensors = (self.tensors[0], torch.Tensor(labels))


# 负责根据各种分布生成训练集
class DatasetController:
    def __init__(self, dataset_name=config.DATASET_NAME):
        dataset_name = dataset_name.upper()
        # get dataset from torchvision.datasets if exists
        if hasattr(torchvision.datasets, dataset_name):
            # set transformation differently per dataset
            if dataset_name in ["CIFAR10"]:
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
            elif dataset_name in ["MNIST"]:
                transform = torchvision.transforms.ToTensor()

            # prepare raw training & test datasets
            training_dataset = torchvision.datasets.__dict__[dataset_name](
                root=config.DATA_PATH,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = torchvision.datasets.__dict__[dataset_name](
                root=config.DATA_PATH,
                train=False,
                download=True,
                transform=transform
            )
        else:
            # dataset not found exception
            error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
            raise AttributeError(error_message)

        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        self.num_class = np.unique(training_dataset.targets).shape[0]

        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
            test_dataset.data = np.asarray(test_dataset.data)

        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()
            test_dataset.targets = test_dataset.targets.tolist()

        data = np.concatenate((training_dataset.data, test_dataset.data))
        targets = []
        targets.extend(training_dataset.targets)
        targets.extend(test_dataset.targets)

        sorted_data, sorted_indices = self.get_sorted_data_and_indices(data, targets)

        self.sorted_data = sorted_data
        self.sorted_indices = sorted_indices
        self.transform = transform

    def get_sorted_data_and_indices(self, data, targets):
        if type(targets) is not type(torch.tensor([5])):
            targets = torch.tensor(targets)

        sorted_targets = torch.sort(targets)[0]
        sorted_indices = torch.sort(targets)[1]
        data = data[sorted_indices]

        dataset_indices = []
        start = 0
        for count in torch.bincount(sorted_targets):
            dataset_indices.append(np.arange(start, start + count))
            start += count

        return data, dataset_indices

    def draw_data_by_distribution(self, distribution, total_samples, remove_from_pool=True, draw_from_pool=True):
        assert (len(self.sorted_indices), len(distribution))

        distribution = distribution * total_samples
        new_input = []
        new_label = []
        for class_id, n in enumerate(distribution):
            selected_indices = self.draw_data_index_by_class(class_id, n, remove_from_pool, draw_from_pool)
            new_input.extend(self.sorted_data[selected_indices])

            extra = max(int(n) - len(selected_indices), 0)
            if extra > 0:
                extra_indices = self.draw_data_index_by_class(class_id, extra, remove_from_pool=False,
                                                              draw_from_pool=False)
                new_input.extend(self.sorted_data[extra_indices])

            for _ in range(int(n)):
                new_label.append(class_id)

        new_dataset = CustomTensorDataset((torch.Tensor(np.array(new_input)), torch.Tensor(new_label)),
                                          transform=self.transform)

        return new_dataset

    def draw_data_index_by_class(self, class_id, n, remove_from_pool=True, draw_from_pool=True):
        available_indices = self.sorted_indices[class_id]

        selected_indices = []
        if draw_from_pool:
            indices = np.arange(1, len(available_indices) - 1)
            selected_indices = np.random.choice(indices, min(len(indices), int(n)), replace=False)
        else:
            indices = np.arange(available_indices[0], available_indices[-1])
            selected_indices = np.random.choice(indices, int(n), replace=False)

        if remove_from_pool:
            self.sorted_indices[class_id] = np.delete(available_indices, selected_indices)

        return selected_indices

    def update_clients_datasets(self, clients, n):
        for client in clients:
            new_train_set = self.draw_data_by_distribution(client.distribution, n)
            client.update_train(new_train_set, replace=False)
            new_test_set = self.draw_data_by_distribution(client.distribution, n * config.TRAIN_TEST_SPLIT,
                                                          remove_from_pool=False, draw_from_pool=False)
            client.update_test(new_test_set, replace=True)
            # client.train.class_swap(1, 2)


def get_dataloader(data):
    return DataLoader(data, batch_size=data.batch_size, shuffle=True)