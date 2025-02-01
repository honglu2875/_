import dataclasses
from typing import Any
import os
import ray
from ft.ray.ray_utils import ray_noset_visible_devices


@dataclasses.dataclass(frozen=True, slots=True)
class DistInfo:  # metadata about the current process in the cluster
    world_size: int
    rank: int
    master_addr: str
    master_port: int

    def set_env_var(self):
        """Set environment variables according to the metadata. It does globally."""
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

    def replace(self, key: str, val: Any) -> "DistInfo":
        """'Mutate' the object by returning a new one with replaced key for convenience."""
        args = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        assert key in args
        args[key] = val
        return DistInfo(**args)

