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
    """
    Executes a function on rank 0 and broadcasts its result to all other ranks.
    Args:
        n (int): Number of objects expected in the list returned by `function` if it returns a list.
                 If `function` returns a single object, n should be 1.
        function (callable): The function to execute on rank 0.
        *args: Positional arguments for `function`.
        **kwargs: Keyword arguments for `function`.
    Returns:
        The result of the function, broadcasted to all ranks.
    """
    if torch.distributed.is_initialized():
        backend = torch.distributed.get_backend()

        object_list_for_broadcast = None  # Initialize to avoid linter warnings

        if torch.distributed.get_rank() == 0:
            # Rank 0 executes the function
            output_from_function = function(*args, **kwargs)

            if n == 1:
                # Expecting a single item. Wrap it in a list for broadcast.
                object_list_for_broadcast = [output_from_function]
            else:  # n > 1
                # Expecting 'n' items. 'output_from_function' should be an iterable (list or tuple) of 'n' items.
                if not isinstance(output_from_function, (list, tuple)):
                    raise TypeError(
                        f"Unsloth (distributed_function rank 0): Expected a list or tuple of {n} items "
                        f"from the wrapped function '{function.__name__}', but got type {type(output_from_function)} "
                        f"with value: {output_from_function}."
                    )

                # Convert to list for broadcast_object_list and ensure it has 'n' items.
                object_list_for_broadcast = list(output_from_function)

                if len(object_list_for_broadcast) != n:
                    raise ValueError(
                        f"Unsloth (distributed_function rank 0): Expected {n} items from the wrapped function "
                        f"'{function.__name__}', but got {len(object_list_for_broadcast)} items. "
                        f"Output was: {output_from_function}"
                    )
            print(
                f"[Rank 0 DEBUG] distributed_function: n={n}, type(output_from_function)={type(output_from_function)}, output_from_function={output_from_function}, len(object_list_for_broadcast)={len(object_list_for_broadcast)}, object_list_for_broadcast={object_list_for_broadcast}",
                flush=True)
        else:
            # Other ranks prepare a list of Nones to receive the broadcasted objects
            object_list_for_broadcast = [None for _ in range(n)]

        # Determine the device for broadcast_object_list
        if backend == "nccl":
            # Ensure CUDA is available and initialized for the current process
            if not torch.cuda.is_available():
                raise RuntimeError("Unsloth (distributed_function): NCCL backend selected, but CUDA is not available.")
            if not torch.cuda.is_initialized():  # Should be initialized by DDP setup
                torch.cuda.init()  # Initialize CUDA for the current process if not already
            comm_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:  # For "gloo", "mpi", or other backends that might prefer CPU
            comm_device = torch.device("cpu")

        # Perform the broadcast operation
        torch.distributed.broadcast_object_list(object_list_for_broadcast, src=0, group=None, device=comm_device)

        # object_list_for_broadcast now contains the results on all ranks.
        # If n=1, it's like [value]. If n>1, it's like [value1, value2, ...].

        if n == 1:
            final_result = object_list_for_broadcast[0]
        else:  # n > 1
            # The caller expects 'n' items, usually for unpacking.
            # object_list_for_broadcast is already a list of these 'n' items.
            final_result = object_list_for_broadcast

    else:  # Not in a distributed environment
        # Execute the function directly. The return structure (single item or tuple/list for n>1)
        # is determined by the function itself.
        final_result = function(*args, **kwargs)

    return final_result

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
