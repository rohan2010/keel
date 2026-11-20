"""BFCL-style function parameters -> JSON Schema (pure Python; used by the BFCL and When2Call evals)."""

_TYPE = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def to_json_schema(s):
    if isinstance(s, dict):
        out = {k: to_json_schema(v) for k, v in s.items()}
        if isinstance(out.get("type"), str):
            out["type"] = _TYPE.get(out["type"], out["type"])
        return out
    if isinstance(s, list):
        return [to_json_schema(x) for x in s]
    return s
