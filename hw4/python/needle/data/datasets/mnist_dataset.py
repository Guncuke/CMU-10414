from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows*cols)
            X = X.reshape(num_images, rows, cols, 1)
            X = X.astype(np.float32) / 255.0
    
        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)
        
        self.image = X
        self.label = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.image[index]), self.label[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.image)
        ### END YOUR SOLUTION