"""Parse native OpenAI tool_calls or text/JSON fallbacks (Qwen-friendly)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardweaver.hwa.llm.tool_call_ids_v1 import new_tool_call_id, normalize_tool_call_id


def _new_id() -> str:
    return new_tool_call_id()


def _parse_args(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {k: _coerce_jsonish(v) for k, v in raw.items()}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        if isinstance(val, dict):
            return {k: _coerce_jsonish(v) for k, v in val.items()}
        return {"value": _coerce_jsonish(val)}
    return {"_raw": raw}


def _coerce_jsonish(val: Any, *, depth: int = 0) -> Any:
    """Recursively parse JSON-encoded strings (P0-2 nested type preservation)."""
    if depth > 6:
        return val
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and (
            (s.startswith("{") and s.endswith("}"))
            or (s.startswith("[") and s.endswith("]"))
        ):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return val
            return _coerce_jsonish(parsed, depth=depth + 1)
        return val
    if isinstance(val, dict):
        return {k: _coerce_jsonish(v, depth=depth + 1) for k, v in val.items()}
    if isinstance(val, list):
        return [_coerce_jsonish(v, depth=depth + 1) for v in val]
    return val


def _action_from_obj(obj: Mapping[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
    name = obj.get("name") or obj.get("tool")
    fn = obj.get("function")
    if isinstance(fn, dict) and not name:
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("args") or {}
    else:
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if isinstance(name, dict):
        name = name.get("name")
    if not name:
        return None
    return {
        "id": normalize_tool_call_id(str(obj.get("id") or ""), fallback_index=0) if obj.get("id") else _new_id(),
        "name": str(name),
        "arguments": _parse_args(args),
        "source": source,
    }


def actions_from_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for tc in tool_calls:
        act = _action_from_obj(tc, source="native_tool_calls")
        if act:
            # Prefer nested function name/args when present (OpenAI shape)
            fn = tc.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                act["name"] = str(fn["name"])
                act["arguments"] = _parse_args(fn.get("arguments", tc.get("arguments")))
                act["id"] = normalize_tool_call_id(
                    str(tc.get("id") or act["id"]),
                    fallback_index=len(actions),
                )
            actions.append(act)
    return actions


_XML_TOOL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_OLMO_LINE_CALL_RE = re.compile(r"(?m)^[ \t]*([a-zA-Z_]\w*)\s*\(")
_OLMO_SKIP_NAMES = frozenset(
    {
        "if",
        "for",
        "while",
        "def",
        "class",
        "return",
        "import",
        "from",
        "elif",
        "else",
        "try",
        "except",
        "with",
        "as",
        "pass",
        "raise",
        "lambda",
        "print",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "set",
        "tuple",
    }
)


def _ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_value(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_value(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        out: Dict[Any, Any] = {}
        for key, val in zip(node.keys, node.values):
            if key is None:
                continue
            out[_ast_value(key)] = _ast_value(val)
        return out
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    raise ValueError(f"unsupported ast node: {type(node).__name__}")


def _action_from_ast_call(call: ast.Call, *, source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(call.func, ast.Name):
        return None
    if call.args:
        return None
    name = call.func.id
    if name in _OLMO_SKIP_NAMES:
        return None
    args: Dict[str, Any] = {}
    for kw in call.keywords:
        if not kw.arg:
            continue
        try:
            args[kw.arg] = _ast_value(kw.value)
        except ValueError:
            return None
    return {
        "id": _new_id(),
        "name": name,
        "arguments": _parse_args(args),
        "source": source,
    }


def _extract_balanced_call(text: str, open_paren_idx: int) -> Optional[str]:
    depth = 0
    in_str = False
    quote = ""
    esc = False
    for j in range(open_paren_idx, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx : j + 1]
    return None


def _extract_pythonic_calls(text: str) -> List[str]:
    """Extract Olmo/vLLM pythonic tool call slices: name(kw=val, ...)."""
    body = text
    wrapped = re.search(r"<function_calls>(.*?)</function_calls>", body, re.DOTALL | re.IGNORECASE)
    if wrapped:
        body = wrapped.group(1)
    kept: List[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        kept.append(line)
    body = "\n".join(kept)
    calls: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        m = _OLMO_LINE_CALL_RE.search(body, i)
        if not m:
            break
        name = m.group(1)
        open_paren = m.end() - 1
        if name in _OLMO_SKIP_NAMES:
            i = open_paren + 1
            continue
        segment = _extract_balanced_call(body, open_paren)
        if not segment:
            i = open_paren + 1
            continue
        calls.append(f"{name}{segment}")
        i = open_paren + len(segment)
    return calls


def actions_from_olmo_pythonic_content(content: str) -> List[Dict[str, Any]]:
    """Parse Olmo-3 pythonic newline tool calls when vLLM did not emit native tool_calls."""
    actions: List[Dict[str, Any]] = []
    for blob in _extract_pythonic_calls(content):
        try:
            node = ast.parse(blob, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call):
            continue
        act = _action_from_ast_call(node, source="olmo_pythonic")
        if act:
            actions.append(act)
    return actions


def _extract_balanced_objects(text: str) -> List[str]:
    """Extract top-level `{...}` slices that can contain nested braces."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        in_str = False
        esc = False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : j + 1])
                    i = j + 1
                    break
        else:
            break
    return out


def _looks_like_tool_json(obj: Mapping[str, Any]) -> bool:
    if obj.get("tool_call") or obj.get("type") == "function":
        return True
    return bool(obj.get("name") or obj.get("tool") or obj.get("function"))


def _action_from_olmo_json_obj(obj: Mapping[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
    tool_call = obj.get("tool_call")
    if isinstance(tool_call, Mapping):
        inner = dict(tool_call)
        name = inner.pop("name", None) or inner.pop("tool", None)
        if name:
            return _action_from_obj({"name": name, "arguments": inner}, source=source)
    if obj.get("type") == "function" and isinstance(obj.get("function"), Mapping):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments") or fn.get("properties") or fn.get("parameters") or {}
        if name:
            return _action_from_obj({"name": name, "arguments": args}, source=source)
    return None


def _pick_olmo_json_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(actions) <= 1:
        return actions
    priority = [
        a
        for a in actions
        if str(a.get("name") or "").startswith(("controller_", "run_", "submit_", "list_", "inspect_"))
    ]
    return [priority[0]] if priority else [actions[0]]


def actions_from_olmo_json_content(content: str) -> List[Dict[str, Any]]:
    """Parse Olmo-3 inline JSON tool blobs: {\"tool_call\":{...}} or type=function lines."""
    actions: List[Dict[str, Any]] = []
    for line in content.splitlines():
        s = line.strip()
        if not s.startswith("{") or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        act = _action_from_olmo_json_obj(obj, source="olmo_json_line")
        if act is None:
            act = _action_from_obj(obj, source="olmo_json_line")
        if act:
            actions.append(act)
    return _pick_olmo_json_actions(actions)


def _actions_from_parsed(obj: Any, *, source: str) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
            return actions_from_tool_calls(obj["tool_calls"])
        act = _action_from_olmo_json_obj(obj, source=source)
        if act:
            return [act]
        act = _action_from_obj(obj, source=source)
        if act:
            actions.append(act)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                act = _action_from_olmo_json_obj(item, source=source)
                if act is None:
                    act = _action_from_obj(item, source=source)
                if act:
                    actions.append(act)
    return actions


def actions_from_content(content: Optional[str]) -> List[Dict[str, Any]]:
    """Best-effort parse of JSON / XML / Olmo pythonic tool invocations from assistant text."""
    if not content or not str(content).strip():
        return []
    text = str(content).strip()
    actions: List[Dict[str, Any]] = []

    # Qwen-style <tool_call>{...}</tool_call>
    for match in _XML_TOOL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        actions.extend(_actions_from_parsed(obj, source="xml_tool_call"))
    if actions:
        return actions

    # Olmo-3 inline JSON lines: {"tool_call": {"name": "...", ...}}
    olmo_json = actions_from_olmo_json_content(text)
    if olmo_json:
        return olmo_json

    # Olmo-3 pythonic: controller_commit_route(route_id="...", ...)
    olmo_actions = actions_from_olmo_pythonic_content(text)
    if olmo_actions:
        return olmo_actions

    # Markdown fences: ```json {..} ``` or ``` [{..}] ```
    for fence in _FENCE_RE.finditer(text):
        body = fence.group(1).strip()
        try:
            obj = json.loads(body)
            got = _actions_from_parsed(obj, source="json_fence")
            if got:
                return got
        except json.JSONDecodeError:
            # Maybe multiple objects inside fence — fall through to balanced extractor
            for blob in _extract_balanced_objects(body):
                try:
                    obj = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                actions.extend(_actions_from_parsed(obj, source="json_fence_obj"))
            if actions:
                return actions

    # Whole-message JSON
    try:
        obj = json.loads(text)
        got = _actions_from_parsed(obj, source="json_object")
        if got:
            return got
    except json.JSONDecodeError:
        pass

    # Loose balanced objects embedded in prose (handles nested arguments)
    for blob in _extract_balanced_objects(text):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if not _looks_like_tool_json(obj):
            continue
        actions.extend(_actions_from_parsed(obj, source="json_balanced"))
    return actions


def parse_assistant_message(message: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (actions, parse_error). Empty actions + error means unparsable."""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        actions = actions_from_tool_calls(tool_calls)
        if actions:
            return actions, None
    content = message.get("content")
    actions = actions_from_content(content if isinstance(content, str) else None)
    if actions:
        return actions, None
    if content and str(content).strip():
        return [], "no_tool_calls_in_content"
    return [], "empty_assistant_message"
