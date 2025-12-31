
import torch
from models.beta_vae import BetaVAE

def verify_fix():
    print("Initializing BetaVAE...")
    try:
        model = BetaVAE(in_channels=3, latent_dim=128)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    print("Checking LPIPS model parameters requires_grad status:")
    trainable_params = sum(p.requires_grad for p in model.lpips_model.parameters())
    all_params = sum(1 for p in model.lpips_model.parameters())
    
    print(f"Total LPIPS params: {all_params}")
    print(f"Trainable LPIPS params: {trainable_params}")
    
    if trainable_params == 0:
        print("\nSUCCESS: LPIPS parameters are successfully frozen. The perceptual loss should no longer diverge negatively.")
    else:
        print("\nFAILURE: Some LPIPS parameters are still trainable.")

if __name__ == "__main__":
    verify_fix()
