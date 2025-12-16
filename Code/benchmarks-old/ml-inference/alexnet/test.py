import rmm
from rmm.allocators.torch import rmm_torch_allocator
import torch

if __name__ == "__main__":
    rmm.reinitialize(pool_allocator=True, managed_memory=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    print("success")