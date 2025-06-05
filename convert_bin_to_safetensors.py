import torch
from safetensors.torch import save_file

def convert_bin_to_safetensors(bin_path, safetensors_path):
    # Load the state dict directly
    state_dict = torch.load(bin_path, map_location="cpu")
    
    # The state_dict is already the layer weights, no need to extract
    print(f"Found {len(state_dict)} layers")
    print(f"Sample keys: {list(state_dict.keys())[:3]}")
    
    # Handle shared tensors by cloning them
    shared_tensors = [
        'language_model.model.shared.weight',
        'language_model.model.decoder.embed_tokens.weight', 
        'language_model.model.encoder.embed_tokens.weight',
        'language_model.lm_head.weight'
    ]
    
    # Create new state dict with cloned shared tensors
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if key in shared_tensors:
            new_state_dict[key] = tensor.clone()
        else:
            new_state_dict[key] = tensor
    
    save_file(new_state_dict, safetensors_path)
    print(f"Converted {bin_path} to {safetensors_path}")

if __name__ == "__main__":
    bin_path = "/home/troth/code/hiwi/flower_vla_calvin/pretrained/Florence-2-large/pytorch_model.bin"
    safetensors_path = "/home/troth/code/hiwi/flower_vla_calvin/pretrained/Florence-2-large/model.safetensors"
    convert_bin_to_safetensors(bin_path, safetensors_path)
