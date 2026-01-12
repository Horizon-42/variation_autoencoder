from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List

class CustomImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        """
        Args:
            image_paths (List[str]): List of paths to the images or npy files.
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
        
        # Check if the file is a numpy file
        if img_path.endswith('.npy'):
            # Load numpy array
            image_array = np.load(img_path)
            
            # Convert numpy array to PIL Image
            # Handle different array shapes and data types
            if image_array.dtype != np.uint8:
                # Normalize to [0, 255] if not already
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Handle different array shapes
            if len(image_array.shape) == 2:
                # Grayscale: (H, W) -> (H, W, 3)
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3:
                # Check if shape is (C, H, W) format
                if image_array.shape[0] in [1, 3, 4] and image_array.shape[0] < min(image_array.shape[1], image_array.shape[2]):
                    # Transpose from (C, H, W) to (H, W, C)
                    image_array = np.transpose(image_array, (1, 2, 0))
                
                # Now handle channel dimension (should be last dimension)
                if image_array.shape[2] == 1:
                    # Single channel: (H, W, 1) -> (H, W, 3)
                    image_array = np.repeat(image_array, 3, axis=2)
                elif image_array.shape[2] == 4:
                    # RGBA: remove alpha channel
                    image_array = image_array[:, :, :3]
                elif image_array.shape[2] != 3:
                    # Unexpected number of channels, replicate first channel
                    image_array = np.repeat(image_array[:, :, 0:1], 3, axis=2)
            
            # Convert to PIL Image
            image = Image.fromarray(image_array, mode='RGB')
        else:
            # Load regular image file
            image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image