import dataclasses
import json
from abc import abstractmethod
from typing import Any


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        json_method = None
        if hasattr(obj.__class__, "__json__"):
            json_method = getattr(obj.__class__, "__json__")
        if callable(json_method):
            return json_method(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

    @abstractmethod
    def dump(self, obj: Any, file):
        json.dump(obj, file, cls=JSONEncoder)

    @abstractmethod
    def dumps(self, obj: Any):
        json.dumps(obj, cls=JSONEncoder)
