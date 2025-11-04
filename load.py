import math
import os
from typing import Dict, Tuple
import json

# Try to import the built C++ extensions
try:
    import dtypes.bfloat16 as bf
except ImportError:
    # Fall back to the build directory for development
    import dtypes.build.bfloat16 as bf

# Try to import torch - it's optional
try:
    import torch
    from safetensors.torch import safe_open
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    safe_open = None

    
    
BYTES_PER_BLOCK = 16
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales") for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales") for n in range(36)
}

# Torch-dependent code - only available when torch is installed
if HAS_TORCH:
    class Checkpoint:

        def __init__(self, path: str, device: torch.device):
            device_str = (
                device.type
                if device.index is None
                else device.type + ":" + str(device.index)
            )
            self.device_str = device_str

            # Read from all files ending with .safetensors in the checkpoint directory
            safetensor_files = [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".safetensors")
            ]
            # Build a mapping from tensor name to (file, key)
            tensor_name_to_file = {}
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                    for key in f.keys():
                        tensor_name_to_file[key] = safetensor_file

            self.tensor_name_to_file = tensor_name_to_file

        def get(self, name: str) -> torch.Tensor:
            match PARAM_NAME_MAP.get(name, name):
                case (blocks_name, scales_name):
                    # MoE weights: are in block-based MXFP4 format
                    return self._get_mxfp4_tensor(blocks_name, scales_name, dtype=torch.bfloat16)
                case tensor_name:
                    # MoE biases and other weights
                    return self._get_tensor(tensor_name)

        def _get_tensor(self, name: str) -> str:
            assert name in self.tensor_name_to_file, f"Tensor {name} not found in checkpoint."
            with safe_open(
                self.tensor_name_to_file[name], framework="pt", device=self.device_str
            ) as f:
                return f.get_tensor(name)

        def _get_mxfp4_tensor(
            self,
            blocks_name: str,
            scales_name: str,
            *,
            dtype: torch.dtype = torch.bfloat16,
            rows_per_chunk: int = 16384 * 512,
        ) -> torch.Tensor:
            
            assert blocks_name in self.tensor_name_to_file, (
                f"Blocks tensor {blocks_name} not found in checkpoint."
            )
            assert scales_name in self.tensor_name_to_file, (
                f"Scales tensor {scales_name} not found in checkpoint."
            )

            blocks = self._get_tensor(blocks_name)
            scales = self._get_tensor(scales_name).to(torch.int32) - 127

            assert blocks.shape[:-1] == scales.shape, (
                f"{blocks.shape=} does not match {scales.shape=}"
            )

            lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

            *prefix_shape, G, B = blocks.shape
            rows_total   = math.prod(prefix_shape) * G

            blocks = blocks.reshape(rows_total, B)
            scales = scales.reshape(rows_total, 1)

            out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

            for r0 in range(0, rows_total, rows_per_chunk):
                r1 = min(r0 + rows_per_chunk, rows_total)

                blk = blocks[r0:r1]
                exp = scales[r0:r1]

                # nibble indices -> int64
                idx_lo = (blk & 0x0F).to(torch.long)
                idx_hi = (blk >> 4).to(torch.long)

                sub = out[r0:r1]
                sub[:, 0::2] = lut[idx_lo]
                sub[:, 1::2] = lut[idx_hi]

                torch.ldexp(sub, exp, out=sub)
                del idx_lo, idx_hi, blk, exp

            return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


    def load_state_dict(
        path: str, config, device: str | torch.device = "cpu", load_in_tensors=False
    ) -> Dict[str, torch.Tensor]:
        """
        Loads a model's configuration and state_dict from a checkpoint directory.

        This function is decoupled from any specific model class.

        Args:
            path: The path to the checkpoint directory.
            device: The device to load the tensors onto.
            load_in_tensors: If True, loads in torch tensors, if false, load in our Array format.

        Returns:
            A tuple containing:
            - The model configuration (ModelConfig).
            - The state dictionary (dict[str, torch.Tensor]).
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Get distributed training info if available
        my_rank =  0
        world_size =  1
        per_rank_intermediate_size = config.intermediate_size // world_size

        checkpoint = Checkpoint(path, device)
        state_dict = {}

        # Discover model parameter names by inspecting checkpoint files
        model_param_names = set()
        for raw_name in checkpoint.tensor_name_to_file.keys():
            if raw_name.endswith((".blocks", ".scales")):
                base_name = ".".join(raw_name.split('.')[:-1])
                model_param_names.add(base_name)
            else:
                model_param_names.add(raw_name)

        # Load, shard, and add each parameter to the state_dict
        for name in sorted(list(model_param_names)):
            loaded_tensor = checkpoint.get(name)
            print(name)
            # Shard the loaded tensor for tensor parallelism if necessary
            if "mlp1" in name:  # both weight and bias
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * 2 * per_rank_intermediate_size : (my_rank + 1) * 2 * per_rank_intermediate_size,
                    ...,
                ]
            elif "mlp2_weight" in name:  # only weight
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank * per_rank_intermediate_size : (my_rank + 1) * per_rank_intermediate_size,
                ]
            
            if load_in_tensors:
                state_dict[name] = loaded_tensor
            else:
                state_dict[name] = bf.Array(loaded_tensor.float().numpy())
                # Free the torch tensor after conversion
                del loaded_tensor

        return state_dict

else:
    # Dummy definitions when torch is not available
    Checkpoint = None
    load_state_dict = None


# Helper function to automatically choose the best loader
def load_checkpoint(path: str, config=None):
    """
    Automatically chooses the best checkpoint loader based on available dependencies.
    
    Args:
        path: Path to checkpoint directory or safetensors file
        config: Model configuration (required for load_state_dict)
    
    Returns:
        Dictionary of tensors in bf.Array format
    """
    if HAS_TORCH and load_state_dict is not None:
        if config is None:
            raise ValueError("config is required when using torch-based loader")
        print("Torch is installed. Using torch-based loader from OpenAI's official implementation (load_state_dict)...")
        return load_state_dict(path, config)
    else:
        print("Using our own checkpoint loader (load_safetensors)...")
        # If path is a directory, assume model.safetensors
        if os.path.isdir(path):
            path = os.path.join(path, "model.safetensors")
        return load_safetensors(path)


def load_safetensors(file_path):
    import numpy as np
    import struct

    dtypes = {
        'U8': np.uint8,
        'BF16': np.dtype('>u2'),
    }

    with open(file_path, 'rb') as f:
        # Read header length (8 bytes)
        header_len = struct.unpack('<Q', f.read(8))[0]

        # Read and parse header
        header = json.loads(f.read(header_len).decode())

        # Load raw tensors first
        raw_tensors = {}
        data_start = 8 + header_len

        for name, info in header.items():
            if name == 'metadata':
                continue

            # Get tensor info
            dtype = dtypes[info['dtype']]
            shape = info['shape']
            start, end = info['data_offsets']

            # Read tensor data
            f.seek(data_start + start)
            data = f.read(end - start)

            # Create tensor
            if info['dtype'] == 'BF16':
                # Convert bfloat16 to float32 (stored as little-endian uint16)
                tensor = np.frombuffer(data, dtype='<u2').reshape(shape)
            else:
                tensor = np.frombuffer(data, dtype=dtype).reshape(shape)

            raw_tensors[name] = tensor

    # Now process tensors, handling MXFP4 format
    tensors = {}
    processed_names = set()

    for name in list(raw_tensors.keys()):
        if name in processed_names:
            continue

        # Check if this is a .blocks tensor with corresponding .scales
        if name.endswith('.blocks'):
            base_name = name[:-7]  # Remove '.blocks'
            scales_name = base_name + '.scales'
            
            if scales_name in raw_tensors:
                # This is an MXFP4 tensor - dequantize it
                blocks = raw_tensors[name]
                scales = raw_tensors[scales_name].astype(np.int32) - 127
                
                dequantized = _dequantize_mxfp4_numpy(blocks, scales)
                tensors[base_name] = bf.Array(dequantized)
                
                # Free memory immediately after processing
                del raw_tensors[name], raw_tensors[scales_name], blocks, scales, dequantized
                
                processed_names.add(name)
                processed_names.add(scales_name)
            else:
                # Just a regular tensor that happens to end with .blocks
                tensors[name] = bf.Array(raw_tensors[name])
                del raw_tensors[name]
                processed_names.add(name)
        elif name.endswith('.scales'):
            # Check if this has a corresponding .blocks
            base_name = name[:-7]  # Remove '.scales'
            blocks_name = base_name + '.blocks'
            
            if blocks_name not in raw_tensors:
                # Standalone .scales tensor (shouldn't happen, but handle it)
                tensors[name] = bf.Array(raw_tensors[name])
                del raw_tensors[name]
                processed_names.add(name)
            # else: will be processed when we encounter the .blocks tensor
        else:
            # Regular tensor
            tensors[name] = bf.Array(raw_tensors[name])
            del raw_tensors[name]
            processed_names.add(name)

    # Clear any remaining references
    del raw_tensors

    return tensors


def _dequantize_mxfp4_numpy(blocks, scales, rows_per_chunk=16384 * 512):
    """
    Dequantizes MXFP4 format tensors using NumPy (no PyTorch).
    
    Args:
        blocks: uint8 array where each byte contains two 4-bit values
        scales: int32 array containing exponents (already adjusted by -127)
        rows_per_chunk: Number of rows to process at once
    
    Returns:
        Float32 array with dequantized values
    """
    import numpy as np
    
    # Create lookup table for FP4 mantissa values
    lut = np.array(FP4_VALUES, dtype=np.float32)
    
    # Get shape information
    *prefix_shape, G, B = blocks.shape
    rows_total = np.prod(prefix_shape, dtype=int) * G
    
    # Reshape for processing
    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)
    
    # Allocate output
    out = np.empty((rows_total, B * 2), dtype=np.float32)
    
    # Process in chunks to manage memory
    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        
        # Extract low and high nibbles (4-bit values from each byte)
        idx_lo = (blk & 0x0F).astype(np.int64)
        idx_hi = (blk >> 4).astype(np.int64)
        
        # Look up mantissa values
        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        
        # Apply exponent scaling: mantissa * 2^exponent
        np.ldexp(sub, exp, out=sub)
        
        # Free memory from this chunk
        del idx_lo, idx_hi, blk, exp
    
    # Reshape back to original dimensions
    return out.reshape(*prefix_shape, G, B * 2).reshape(*prefix_shape, G * B * 2)