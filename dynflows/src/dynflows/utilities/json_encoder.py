import dataclasses
import json
from abc import abstractmethod
from typing import IO, Any


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        json_method = None
        if hasattr(obj.__class__, "__json__"):
            json_method = getattr(obj.__class__, "__json__")
        if callable(json_method):
            return json_method(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return super().default(obj)

    def dump(self, obj: Any, file: IO[str]) -> None:
        json.dump(obj, file, cls=JSONEncoder)

    def dumps(self, obj: Any) -> str:
        return json.dumps(obj, cls=JSONEncoder)
