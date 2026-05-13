from huggingface_hub import hf_hub_download
import zipfile
import os
from pathlib import Path




def read_image(index):
    default_path = Path(__file__).resolve().parents[1] / "resnet" / "cifar10" / "test_batch.bin"
    filePath = os.environ.get("EASYFHE_CIFAR10_TEST_BATCH", str(default_path))
    IMAGE_SIZE = 3072
    LABEL_SIZE = 1
    RECORD_SIZE = LABEL_SIZE + IMAGE_SIZE
    try:
        with open(filePath, "rb") as file:
            file.seek(index * RECORD_SIZE)
            label = file.read(LABEL_SIZE)
            if not label:
                raise ValueError("Failed to read label.")
            label = int.from_bytes(label, byteorder="big")
            # print(f"Label: {label}")
            image_data = file.read(IMAGE_SIZE)
            if not image_data or len(image_data) != 3072:
                raise ValueError("Failed to read image data.")
        imageVector = []
        for channel in range(3):
            for i in range(1024):
                pixel = float(image_data[channel * 1024 + i]) / 255.0
                if channel == 0:
                    pixel = (pixel - 0.4914) / 0.2023
                elif channel == 1:
                    pixel = (pixel - 0.4822) / 0.1994
                elif channel == 2:
                    pixel = (pixel - 0.4465) / 0.2010
                imageVector.append(pixel)
        return imageVector, label, index
    except FileNotFoundError:
        print(f"Failed to open the file: {filePath}")
