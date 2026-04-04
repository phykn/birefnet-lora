from typing import Any


class ConfigDict(dict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self[name] = value

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output
