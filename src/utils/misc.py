from typing import Any


class ConfigDict(dict):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(
        self,
        name: str,
        value: Any
    ) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

