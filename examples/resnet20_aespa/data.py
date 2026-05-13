import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_TEST_BATCH = DEFAULT_DATA_DIR / "cifar10" / "test_batch.bin"

IMAGE_SIZE = 3072
LABEL_SIZE = 1
RECORD_SIZE = LABEL_SIZE + IMAGE_SIZE
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def resolve_test_batch_path(data_dir=None, test_batch_path=None):
    if test_batch_path is None:
        test_batch_path = os.environ.get("EASYFHE_CIFAR10_TEST_BATCH")
    if test_batch_path is not None:
        return Path(test_batch_path)

    if data_dir is None:
        data_dir = os.environ.get("EASYFHE_RESNET20_AESPA_DATA_DIR", DEFAULT_DATA_DIR)
    return Path(data_dir) / "cifar10" / "test_batch.bin"


def read_image(index, data_dir=None, test_batch_path=None):
    file_path = resolve_test_batch_path(data_dir, test_batch_path)
    with open(file_path, "rb") as file:
        file.seek(index * RECORD_SIZE)
        label_data = file.read(LABEL_SIZE)
        if not label_data:
            raise ValueError(f"Failed to read CIFAR-10 label at index {index} from {file_path}")
        label = int.from_bytes(label_data, byteorder="big")

        image_data = file.read(IMAGE_SIZE)
        if len(image_data) != IMAGE_SIZE:
            raise ValueError(f"Failed to read CIFAR-10 image at index {index} from {file_path}")

    image_vector = []
    for channel, (mean, std) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        channel_offset = channel * 1024
        for pixel_index in range(1024):
            pixel = float(image_data[channel_offset + pixel_index]) / 255.0
            image_vector.append((pixel - mean) / std)

    return image_vector, label, index
