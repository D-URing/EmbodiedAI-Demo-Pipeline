from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from embodied_demo.errors import ConfigurationError, SchemaValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file not found: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"top-level YAML value must be a mapping: {path}")
    return payload


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; scalars and lists are replaced by the override."""

    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def compose_yaml(path: str | Path) -> tuple[dict[str, Any], list[Path]]:
    """Compose a YAML file and its ``extends`` chain in deterministic order."""

    root = Path(path).expanduser().resolve()

    def visit(current: Path, stack: tuple[Path, ...]) -> tuple[dict[str, Any], list[Path]]:
        if current in stack:
            cycle = " -> ".join(str(item) for item in (*stack, current))
            raise ConfigurationError(f"configuration extends cycle detected: {cycle}")

        raw = _read_yaml(current)
        extends = raw.pop("extends", [])
        if isinstance(extends, str):
            extends = [extends]
        if not isinstance(extends, list) or not all(isinstance(item, str) for item in extends):
            raise ConfigurationError(
                f"'extends' must be a string or list of strings: {current}"
            )

        merged: dict[str, Any] = {}
        sources: list[Path] = []
        for reference in extends:
            parent_path = (current.parent / reference).resolve()
            parent_data, parent_sources = visit(parent_path, (*stack, current))
            merged = _deep_merge(merged, parent_data)
            sources.extend(parent_sources)

        merged = _deep_merge(merged, raw)
        sources.append(current)
        return merged, list(dict.fromkeys(sources))

    return visit(root, ())


def _validate(model: type[ModelT], payload: dict[str, Any], source: Path) -> ModelT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise SchemaValidationError(f"schema validation failed for {source}:\n{exc}") from exc


def dump_yaml(payload: BaseModel | Mapping[str, Any]) -> str:
    if isinstance(payload, BaseModel):
        serializable = payload.model_dump(mode="json")
    else:
        serializable = dict(payload)
    return yaml.safe_dump(
        serializable,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
