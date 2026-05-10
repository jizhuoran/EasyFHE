class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0]) if self.tensors else 0


class DataLoader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torch.utils.data is not available in EasyFHE")


def random_split(*args, **kwargs):
    raise RuntimeError("torch.utils.data is not available in EasyFHE")


def default_collate(batch):
    return batch


def get_worker_info():
    return None


__all__ = [
    "DataLoader",
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "default_collate",
    "get_worker_info",
    "random_split",
]

