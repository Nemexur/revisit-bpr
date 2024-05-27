from typing import Any, Callable, Optional
from abc import ABC, abstractmethod
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Launcher(ABC):
    @abstractmethod
    def spawn(self, fn: Callable, *args):
        pass

    @abstractmethod
    def init_env(self, local_rank: int, world_size: int) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass


class Simple(Launcher):
    def spawn(self, fn: Callable, *args):
        return fn(*args)

    def init_env(self, *_) -> None:
        pass

    def clean(self) -> None:
        pass


class DDP(Launcher):
    def __init__(
        self,
        address: str = "127.0.0.1",
        backend: str = "nccl",
        port: str | int = 2112,
        world_size: Optional[int] = None,
        process_group_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._address = os.environ.get("MASTER_ADDR", address)
        self._port = os.environ.get("MASTER_PORT", str(port))
        self._backend = backend
        self._world_size = world_size
        self._process_group_kwargs = process_group_kwargs or {}

    def spawn(self, fn: Callable, *args):
        world_size = self._world_size or torch.cuda.device_count()
        return mp.spawn(
            fn,
            args=(world_size, *args),
            nprocs=world_size,
            join=True,
        )

    def init_env(self, local_rank: int, world_size: int) -> None:
        process_group_kwargs = {
            "backend": self._backend,
            "world_size": world_size,
            **self._process_group_kwargs,
        }
        os.environ["MASTER_ADDR"] = str(self._address)
        os.environ["MASTER_PORT"] = str(self._port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(**process_group_kwargs)

    def clean(self) -> None:
        dist.destroy_process_group()
