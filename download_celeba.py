import torchvision
import os

def download_celeba(root='./data'):
    """
    Downloads the CelebA dataset using torchvision.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Created directory: {root}")
    
    print(f"Attempting to download CelebA dataset to: {root}")
    
    try:
        # download=True will download the dataset if it's not already there
        # split='all' downloads all splits (train, valid, test)
        # target_type='attr' is the default, getting attributes
        dataset = torchvision.datasets.CelebA(
            root=root, 
            split='all', 
            target_type='attr', 
            download=True
        )
        print("Successfully downloaded/verified CelebA dataset.")
        print(f"Dataset size: {len(dataset)}")
        
    except Exception as e:
        print(f"Failed to download CelebA: {e}")
        print("Note: Sometimes automatic download fails due to Google Drive quotas.")
        print("If that happens, you may need to download it manually from the official website.")

if __name__ == "__main__":
    # You can change the root to wherever you want the data stored
    # Using specific path inside the current workspace
    download_celeba(root='./data')
