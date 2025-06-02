# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "Version",
    "_get_dtype",
    "is_main_process",
    "is_distributed",
    "distributed_function",
]

from packaging.version import Version as TrueVersion
import torch
import torch.distributed as dist
import time

def Version(version):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        return TrueVersion(version)
    except:
        from inspect import getframeinfo, stack
        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n" \
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )
    pass


pass

__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}


def _get_dtype(dtype):
    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            try:
                dtype = eval(f"torch.{dtype.lower()}")
            except:
                pass
        if type(dtype) is torch.dtype: return dtype
    return None


pass


def is_main_process():
    is_initialized = torch.distributed.is_initialized()
    return (not is_initialized) or (is_initialized and torch.distributed.get_rank() == 0)


pass


def is_distributed():
    return torch.distributed.is_initialized()


pass


# def distributed_function(n = 1, function = None, *args, **kwargs):
#     if torch.distributed.is_initialized():
#         if torch.distributed.get_rank() == 0:
#             object_list = function(*args, **kwargs)
#             if n == 1: object_list = [object_list]
#         else:
#             object_list = [None for _ in range(n)]
#         # broadcast_object_list auto blocks so no need for barrier
#         torch.distributed.broadcast_object_list(object_list, src = 0, device = "cpu")
#         if n == 1: result = object_list[0]
#     else:
#         result = function(*args, **kwargs)
#     return result
# pass

#
def distributed_function(n=1, function=None, *args, **kwargs):
    # If we are not in a distributed context yet, just run locally
    if function is None or not callable(function):
        raise ValueError("distributed_function requires a callable `func` as its second argument")

    # ──────────────────────────
    # 1. Single-process fallback
    # ──────────────────────────
    if not (dist.is_available() and dist.is_initialized()):
        result = function(*args, **kwargs)
        return result if n == 1 else tuple(result)

    # ──────────────────────────
    # 2. Real multi-GPU run
    # ──────────────────────────
    rank = dist.get_rank()
    backend = dist.get_backend()

    # NCCL can only broadcast CUDA tensors → choose device dynamically
    if backend == "nccl":
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:  # gloo, mpi, etc. work with CPU tensors
        device = torch.device("cpu")

    # Rank-0 executes the function
    if rank == 0:
        result = function(*args, **kwargs)
        obj_list = [result] if n == 1 else list(result)
    else:
        result = None
        obj_list = [None] * n

    # Broadcast the object list so every rank gets rank-0’s result(s)
    dist.broadcast_object_list(obj_list, src=0, device=device)

    # Non-zero ranks pick up the broadcasted value
    if rank != 0:
        result = obj_list[0] if n == 1 else tuple(obj_list)

    return result
pass


# def distributed_function(n=1, function=None, *args, **kwargs):
#     """
#     Robust distributed function with compilation support
#     """
#     if function is None or not callable(function):
#         raise ValueError("distributed_function requires a callable `function`")
#
#     # Single-process fallback
#     if not (dist.is_available() and dist.is_initialized()):
#         result = function(*args, **kwargs)
#         return result if n == 1 else (result if n > 1 else result)
#
#     # Multi-GPU distributed execution with enhanced error handling
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#
#     max_retries = 3
#     retry_delay = 1.0
#
#     for attempt in range(max_retries):
#         try:
#             # Synchronize before execution
#             dist.barrier()
#
#             # Only rank 0 executes the function
#             if rank == 0:
#                 result = function(*args, **kwargs)
#                 obj_list = [result] if n == 1 else list(result) if hasattr(result, '__iter__') else [result]
#
#                 # Ensure obj_list has correct length
#                 while len(obj_list) < n:
#                     obj_list.append(None)
#                 obj_list = obj_list[:n]
#             else:
#                 result = None
#                 obj_list = [None] * n
#
#             # Use CPU tensors for broadcasting to avoid NCCL issues
#             try:
#                 # Try NCCL first if available
#                 if dist.get_backend() == "nccl":
#                     # For NCCL, we need to be more careful with object broadcasting
#                     import pickle
#                     import io
#
#                     if rank == 0:
#                         # Serialize the objects
#                         buffer = io.BytesIO()
#                         pickle.dump(obj_list, buffer)
#                         data = buffer.getvalue()
#                         size_tensor = torch.tensor([len(data)], dtype=torch.long, device=f"cuda:{rank}")
#                     else:
#                         size_tensor = torch.tensor([0], dtype=torch.long, device=f"cuda:{rank}")
#
#                     # Broadcast size first
#                     dist.broadcast(size_tensor, src=0)
#
#                     if rank != 0:
#                         data = bytearray(size_tensor.item())
#
#                     # Convert to tensor for broadcasting
#                     data_tensor = torch.tensor(list(data), dtype=torch.uint8, device=f"cuda:{rank}")
#                     dist.broadcast(data_tensor, src=0)
#
#                     if rank != 0:
#                         # Deserialize
#                         buffer = io.BytesIO(bytes(data_tensor.cpu().numpy()))
#                         obj_list = pickle.load(buffer)
#                         result = obj_list[0] if n == 1 else tuple(obj_list) if n > 1 else obj_list
#                 else:
#                     # Fallback to CPU broadcasting
#                     dist.broadcast_object_list(obj_list, src=0, device="cpu")
#                     if rank != 0:
#                         result = obj_list[0] if n == 1 else tuple(obj_list) if n > 1 else obj_list
#
#             except Exception as broadcast_error:
#                 print(f"Rank {rank}: Broadcast failed: {broadcast_error}, falling back to independent execution")
#                 # Fallback: each rank executes independently
#                 result = function(*args, **kwargs)
#                 return result if n == 1 else (result if n > 1 else result)
#
#             # Final synchronization
#             dist.barrier()
#             return result
#
#         except Exception as e:
#             print(f"Rank {rank}: Attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay * (attempt + 1))
#                 continue
#             else:
#                 # Final fallback: independent execution
#                 print(f"Rank {rank}: All attempts failed, executing independently")
#                 result = function(*args, **kwargs)
#                 return result if n == 1 else (result if n > 1 else result)

pass
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
