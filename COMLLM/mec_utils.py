"""Shared helpers for MEC prompt parsing and result normalization."""

import json
import re


INPUT_MARKER = "**输入数据**："


def parse_model_prediction(raw_output):
    """Parse a model output string into a normalized offloading action.

    Returns:
        dict | None: ``{"action_id": int, "action_name": str}`` when parsing
        succeeds. ``action_id`` is 0 for local execution and positive for edge
        servers. ``None`` means the output cannot be mapped to a valid action.
    """
    if not isinstance(raw_output, str):
        return None

    clean_output = raw_output.lower().strip()

    # Match server decisions first so phrases such as "不是本地，选择服务器2"
    # are not accidentally parsed as local execution.
    server_match = re.search(r"(?:边缘服务器|服务器|server)\s*(\d+)", clean_output)
    if server_match:
        try:
            server_id = int(server_match.group(1))
        except ValueError:
            return None

        if server_id > 0:
            return {
                "action_id": server_id,
                "action_name": f"卸载到边缘服务器{server_id}",
            }

    if "本地" in clean_output or "local" in clean_output:
        return {"action_id": 0, "action_name": "本地执行"}

    return None


def extract_input_data_from_prompt(prompt):
    """Extract the serialized MEC input dictionary from a training prompt."""
    if not isinstance(prompt, str):
        return {}

    match = re.search(re.escape(INPUT_MARKER) + r"(.*)", prompt, re.DOTALL)
    if not match:
        return {}

    input_data = {}
    for line in match.group(1).strip().splitlines():
        if ":" not in line:
            continue
        key, value_str = line.split(":", 1)
        key = key.strip()
        value_str = value_str.strip()

        if (value_str.startswith("[") and value_str.endswith("]")) or (
            value_str.startswith("{") and value_str.endswith("}")
        ):
            try:
                value = json.loads(value_str)
            except json.JSONDecodeError:
                value = value_str
        else:
            value = value_str

        input_data[key] = value

    return input_data


def remove_nulls(obj):
    """Recursively remove ``None`` values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {key: remove_nulls(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, list):
        return [remove_nulls(value) for value in obj if value is not None]
    return obj


def format_mec_input(input_data):
    """Serialize an MEC input dictionary as key-value prompt lines."""
    lines = []
    for key, value in input_data.items():
        if isinstance(value, (list, dict)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = value
        lines.append(f"{key}: {value_text}")
    return "\n".join(lines)
