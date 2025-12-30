
import json

notebook_path = "/home/supercomputing/studys/variation_autoencoder/vae_on_celebs.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "optimizer = optim.Adam(vanilla_vae.parameters(), lr=1e-3)" in source:
            new_source = source.replace("optimizer = optim.Adam(vanilla_vae.parameters(), lr=1e-3)", 
                                        "optimizer = optim.Adam(model.parameters(), lr=1e-3)")
            cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.splitlines()]
            # Fix: splitlines() removes the trailing newline which json expects as part of the list elements if we want to be safe
            # But the most standard way is keeping them in the list.
            lines = new_source.splitlines(keepends=True)
            cell['source'] = lines
            found = True
            print("Found and replaced optimizer initialization.")
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Could not find the target line in the notebook.")
