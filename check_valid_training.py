import os
import re
from pathlib import Path

def parse_result_dir_config(dir_path: str) -> dict:
    RESULT_DIR_RE = re.compile(
        r"^results_(?P<model>.+?)_ImSize(?P<img>\d+)_Lat(?P<lat>\d+)(?P<rest>.*)$"
    )
    LOSS_CHOICES = {"H", "B", "PID", "Cyclical"}

    name = Path(dir_path).name
    match = RESULT_DIR_RE.match(name)
    if not match:
        raise ValueError(f"{name} is not a recognized results directory.")

    tokens = [tok for tok in match["rest"].split("_") if tok]
    loss_type = next((tok for tok in tokens if tok in LOSS_CHOICES), None)
    if not loss_type:
        raise ValueError("Loss type tag is missing in directory name.")
    tokens.remove(loss_type)

    cfg = {
        "model_name": match["model"],
        "image_size": int(match["img"]),
        "latent_dim": int(match["lat"]),
        "loss_type": loss_type,
        "enable_perceptual_loss": False,
        "tvl_weight": 0.0,
        "lpips_weight": 0.0,
    }

    for tok in list(tokens):
        if tok.startswith("TVL"):
            cfg["tvl_weight"] = int(tok[3:]) / 1e4
            tokens.remove(tok)
        elif tok.startswith("LPIPS"):
            cfg["lpips_weight"] = int(tok[5:]) / 10.0
            cfg["enable_perceptual_loss"] = True
            tokens.remove(tok)

    if loss_type == "H":
        beta_tok = next((t for t in tokens if t.startswith("Beta")), None)
        if beta_tok:
            cfg["beta"] = float(beta_tok[4:])
    elif loss_type == "B":
        gamma_tok = next((t for t in tokens if t.startswith("G")), None)
        cap_tok = next((t for t in tokens if t.startswith("C")), None)
        if gamma_tok:
            cfg["gamma"] = float(gamma_tok[1:])
        if cap_tok:
            cfg["max_capacity"] = float(cap_tok[1:])
    elif loss_type == "PID":
        exp_tok = next((t for t in tokens if t.startswith("EXP")), None)
        if exp_tok:
            cfg["exp_kld_loss"] = float(exp_tok[3:])
    elif loss_type == "Cyclical":
        cfg["mode"] = "linear"
        for tok in tokens:
            if tok.startswith("MaxBeta"):
                cfg["max_beta"] = float(tok[7:])
            elif tok.startswith("Ratio"):
                cfg["ratio"] = int(tok[5:]) / 100.0
            elif tok.lower() in {"linear", "sigmoid"}:
                cfg["mode"] = tok.lower()

    return cfg

# check all the dir start with results_ in current directory
if __name__ == "__main__":
    current_dir = os.getcwd()
    failled_dirs = []
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and item.startswith("results_"):
            try:
                config = parse_result_dir_config(item_path)
                # add hidden_dims to config
                config["hidden_dims"] = [32, 64, 128, 256, 512]
                if config["image_size"] >= 128 or config["latent_dim"] >= 256:
                    config["hidden_dims"].append(1024)
                # write config to json file
                import json
                with open(os.path.join(item_path, "hyperparameters.json"), "w") as f:
                    json.dump(config, f, indent=4)
            except ValueError as e:
                failled_dirs.append(item_path)
    if failled_dirs:
        print("The following directories failed to parse:")
        for dir_path in failled_dirs:
            print(f"  {dir_path}")