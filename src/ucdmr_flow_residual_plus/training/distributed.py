from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DistributedState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: object

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(torch_module) -> DistributedState:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if torch_module.cuda.is_available():
        torch_module.cuda.set_device(local_rank)
        device = torch_module.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch_module.device("cpu")
        backend = "gloo"
    if distributed and not torch_module.distributed.is_initialized():
        torch_module.distributed.init_process_group(backend=backend, init_method="env://")
    return DistributedState(distributed=distributed, rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier(torch_module, state: DistributedState) -> None:
    if state.distributed and torch_module.distributed.is_initialized():
        if str(state.device).startswith("cuda"):
            torch_module.distributed.barrier(device_ids=[state.local_rank])
        else:
            torch_module.distributed.barrier()


def cleanup(torch_module) -> None:
    if torch_module.distributed.is_available() and torch_module.distributed.is_initialized():
        torch_module.distributed.destroy_process_group()

