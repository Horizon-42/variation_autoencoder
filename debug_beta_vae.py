
import torch
from models.beta_vae import BetaVAE
from models.vanilla_vae import VanillaVAE

def debug_model():
    print("Debugging BetaVAE...")
    IMG_SIZE = (64, 64)
    LATENT_DIM = 128
    
    # Instantiate BetaVAE
    model = BetaVAE(in_channels=3, 
                    latent_dim=LATENT_DIM, 
                    hidden_dims=[32, 64, 128, 256, 512], 
                    image_size=IMG_SIZE, 
                    beta=1,
                    loss_type='H') # Explicitly setting H for verification
    
    print(f"Loss Type: {model.loss_type}")
    print(f"Beta: {model.beta}")
    print(f"Flat Size: {model.flat_size}")
    print(f"Encoder Output Shape: {model.encoder_output_shape}")
    
    # Dummy Input
    x = torch.randn(2, 3, 64, 64)
    
    # Forward Pass
    try:
        results = model.forward(x)
        recons = results[0]
        mu = results[2]
        log_var = results[3]
        print(f"Input shape: {x.shape}")
        print(f"Recons shape: {recons.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"LogVar shape: {log_var.shape}")
        
        # Check Loss
        loss_dict = model.loss_function(recons, x, mu, log_var, M_N=0.005)
        print("Loss calculation successful")
        print(loss_dict)
        
    except Exception as e:
        print(f"Error during forward/loss: {e}")
        import traceback
        traceback.print_exc()

    print("\n-------------------\n")
    print("Debugging VanillaVAE (Reference)...")
    v_model = VanillaVAE(in_channels=3,
                         latent_dim=LATENT_DIM,
                         hidden_dims=[32, 64, 128, 256, 512],
                         image_size=IMG_SIZE)
    
    print(f"Vanilla Flat Size: {v_model.flat_size}")
    print(f"Vanilla Encoder Output Shape: {v_model.encoder_output_shape}")
    
    v_results = v_model.forward(x)
    print(f"Vanilla Recons shape: {v_results[0].shape}")


if __name__ == "__main__":
    debug_model()
