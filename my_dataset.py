from torch.utils.data import Dataset
from PIL import Image
from typing import List

class CustomImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        """
        Args:
            image_paths (List[str]): List of paths to the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.image_paths = image_paths
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset at the given index."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image