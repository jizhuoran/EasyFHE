from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class RuntimeOptions:
    auto_load_keys: Optional[bool] = None
    rotation_random_mode: str = "fresh"
    rotation_key_limb_limits: Mapping[int, int] = field(default_factory=dict)

    def resolved_auto_load_keys(self, device):
        if self.auto_load_keys is not None:
            return bool(self.auto_load_keys)
        return str(device) == "cuda"
