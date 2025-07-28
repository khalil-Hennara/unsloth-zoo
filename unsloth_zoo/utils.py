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
            f"Unsloth: Could not get version for `{version}`\n"\
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
            try: dtype = eval(f"torch.{dtype.lower()}")
            except: pass
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
