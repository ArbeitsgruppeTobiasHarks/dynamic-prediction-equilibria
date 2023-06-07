from abc import abstractmethod
import json
from typing import Any


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if hasattr(obj.__class__, "__json__"):
            json_method = getattr(obj.__class__, "__json__")
        if callable(json_method):
            return json_method(obj)
        return super().default(obj)

    @abstractmethod
    def dump(obj: Any, file):
        json.dump(obj, file, cls=JSONEncoder)

    @abstractmethod
    def dumps(obj: Any):
        json.dumps(obj, cls=JSONEncoder)
