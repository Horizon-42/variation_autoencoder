
import torch
from models.beta_vae import BetaVAE

def check_lpips_grad():
    model = BetaVAE(in_channels=3, latent_dim=128)
    
    print("Checking LPIPS model parameters requires_grad status:")
    trainable_params = 0
    all_params = 0
    for name, param in model.lpips_model.named_parameters():
        all_params += 1
        if param.requires_grad:
            trainable_params += 1
            print(f"Param {name} requires grad: {param.requires_grad}")
            
    print(f"\nTotal LPIPS params: {all_params}")
    print(f"Trainable LPIPS params: {trainable_params}")
    
    if trainable_params > 0:
        print("\nISSUE CONFIRMED: LPIPS model parameters are trainable.")
    else:
        print("\nLPIPS model parameters are frozen.")

    # Check if calling .train() on parent affects LPIPS
    print("\nSwitching BetaVAE to train mode...")
    model.train()
    print(f"LPIPS model training mode: {model.lpips_model.training}") 
    # Note: .train() doesn't change requires_grad, but enables dropout/batchnorm and indicates intent.
    # The real issue is requires_grad being True by default for nn.Module submodules if not explicitly frozen.

if __name__ == "__main__":
    check_lpips_grad()
