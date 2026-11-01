"""A tiny JSON Schema validator for tool arguments.

Supports only the subset tools actually use: type, properties, required, enum, items, additionalProperties: false,
minimum, maximum, minLength, maxLength. Anything else in a schema is IGNORED, not enforced. For full JSON Schema use a
real library. This exists so the lesson has no dependency, and so both sides of a connection can check arguments cheaply.
"""


def _type_ok(value, type_name: str) -> bool:
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return {"string": isinstance(value, str), "boolean": isinstance(value, bool), "array": isinstance(value, list),
            "object": isinstance(value, dict), "null": value is None}.get(type_name, True)


def validate(instance, schema: dict, path: str = "arguments") -> list[str]:
    """Return a list of human-readable problems. Empty means valid (for the supported subset)."""
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]
        if not any(_type_ok(instance, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(instance).__name__}"]
    problems = []
    if "enum" in schema and instance not in schema["enum"]:
        problems.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                problems.append(f"{path}: missing required property '{name}'")
        for key, value in instance.items():
            if key in props:
                problems += validate(value, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                problems.append(f"{path}: unexpected property '{key}'")
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            problems += validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            problems.append(f"{path}: longer than {schema['maxLength']} characters")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            problems.append(f"{path}: below the minimum of {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            problems.append(f"{path}: above the maximum of {schema['maximum']}")
    return problems
