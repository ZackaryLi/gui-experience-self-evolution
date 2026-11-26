import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Union

from .recovery_schemas import RecoveryContext, ToolObservation


ToolHandler = Callable[
    [Dict[str, Any]],
    Union[Dict[str, Any], Awaitable[Dict[str, Any]]],
]


class ToolValidationError(ValueError):
    pass


class ToolOperation(str, Enum):
    UNKNOWN = "UNKNOWN"
    READ_ONLY = "READ_ONLY"
    EXPERIMENT = "EXPERIMENT"
    SENSITIVE = "SENSITIVE"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler
    timeout_seconds: float = 10.0
    operation: ToolOperation = ToolOperation.UNKNOWN
    requires_intent_scope: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("工具名称必须是非空字符串")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(f"工具 {self.name} 的 description 必须是非空字符串")
        if not isinstance(self.input_schema, dict) or self.input_schema.get("type") != "object":
            raise ValueError(f"工具 {self.name} 的 input_schema 必须是 object Schema")
        if not callable(self.handler):
            raise ValueError(f"工具 {self.name} 的 handler 必须可调用")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError(f"工具 {self.name} 的 timeout_seconds 必须是正数")
        if not isinstance(self.operation, ToolOperation):
            raise ValueError(f"工具 {self.name} 的 operation 必须是 ToolOperation")
        if not isinstance(self.requires_intent_scope, bool):
            raise ValueError(f"工具 {self.name} 的 requires_intent_scope 必须是布尔值")

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "timeout_seconds": self.timeout_seconds,
            "operation": self.operation.value,
            "requires_intent_scope": self.requires_intent_scope,
        }


@dataclass
class ToolRegistry:
    _definitions: Dict[str, ToolDefinition] = field(default_factory=dict)

    def __init__(self, definitions: Optional[Iterable[ToolDefinition]] = None):
        self._definitions = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("只能注册 ToolDefinition")
        if definition.name in self._definitions:
            raise ValueError(f"工具重复注册: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as error:
            raise ToolValidationError(f"未知或当前不可用的工具: {name}") from error

    def describe(self) -> List[Dict[str, Any]]:
        return [definition.describe() for definition in self._definitions.values()]

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
            for definition in self._definitions.values()
        ]

    def validate(self, name: str, tool_input: Dict[str, Any]) -> None:
        definition = self.get(name)
        _validate_schema(tool_input, definition.input_schema, "input")

    async def dispatch(self, name: str, tool_input: Dict[str, Any]) -> ToolObservation:
        definition = self.get(name)
        self.validate(name, tool_input)
        try:
            data = await asyncio.wait_for(
                self._invoke(definition, tool_input),
                timeout=definition.timeout_seconds,
            )
            if not isinstance(data, dict):
                return ToolObservation(
                    tool=name,
                    success=False,
                    error="INVALID_TOOL_RESULT: 工具 Handler 必须返回字典",
                )
            return ToolObservation(tool=name, success=True, data=data)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return ToolObservation(
                tool=name,
                success=False,
                error=(
                    f"TOOL_TIMEOUT: 工具 {name} 执行超过 "
                    f"{definition.timeout_seconds} 秒"
                ),
            )
        except Exception as error:
            return ToolObservation(
                tool=name,
                success=False,
                error=f"TOOL_EXECUTION_FAILED: {type(error).__name__}: {error}",
            )

    @staticmethod
    async def _invoke(
        definition: ToolDefinition,
        tool_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = definition.handler(dict(tool_input))
        if inspect.isawaitable(result):
            result = await result
        return result


def build_default_tool_registry(
    context: RecoveryContext,
    context_provider: Optional[Callable[[], RecoveryContext]] = None,
) -> ToolRegistry:
    provider = context_provider or (lambda: context)
    intent_property = {"type": "string", "const": context.intent_id}
    intent_schema = {
        "type": "object",
        "properties": {"intent_id": intent_property},
        "required": ["intent_id"],
        "additionalProperties": False,
    }

    def get_recovery_context(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "OK", "context": provider().to_dict()}

    def get_diagnosis_report(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        current = provider()
        if current.diagnosis is None:
            return {"status": "NOT_FOUND", "intent_id": current.intent_id}
        return {"status": "OK", "diagnosis": current.diagnosis.to_dict()}

    def query_experiment_history(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        current = provider()
        experiment_id = tool_input.get("experiment_id")
        limit = tool_input.get("limit", 20)
        matches = []
        for item in current.history:
            if not isinstance(item, dict):
                raise ValueError("RecoveryContext.history 的每一项都必须是字典")
            if item.get("intent_id", current.intent_id) != current.intent_id:
                continue
            if experiment_id is not None and item.get("experiment_id") != experiment_id:
                continue
            matches.append(item)
        selected = matches[-limit:]
        return {
            "status": "OK",
            "history": selected,
            "count": len(selected),
        }

    def get_experience_version(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        current = provider()
        experience = current.current_experience
        requested_id = tool_input.get("experience_id")
        if experience is None or (
            requested_id is not None and experience.experience_id != requested_id
        ):
            return {
                "status": "NOT_FOUND",
                "intent_id": current.intent_id,
                "experience_id": requested_id,
            }
        return {"status": "OK", "experience": experience.to_dict()}

    history_schema = {
        "type": "object",
        "properties": {
            "intent_id": intent_property,
            "experiment_id": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["intent_id"],
        "additionalProperties": False,
    }
    experience_schema = {
        "type": "object",
        "properties": {
            "intent_id": intent_property,
            "experience_id": {"type": "string", "minLength": 1},
        },
        "required": ["intent_id"],
        "additionalProperties": False,
    }
    return ToolRegistry([
        ToolDefinition(
            name="get_recovery_context",
            description="读取当前意图的完整 Recovery 上下文，只能查询当前 intent_id。",
            input_schema=intent_schema,
            handler=get_recovery_context,
            operation=ToolOperation.READ_ONLY,
            requires_intent_scope=True,
        ),
        ToolDefinition(
            name="get_diagnosis_report",
            description="读取当前意图的结构化诊断报告；报告不存在时返回 NOT_FOUND。",
            input_schema=intent_schema,
            handler=get_diagnosis_report,
            operation=ToolOperation.READ_ONLY,
            requires_intent_scope=True,
        ),
        ToolDefinition(
            name="query_experiment_history",
            description=(
                "精确查询当前意图的实验历史，可按 experiment_id 过滤；"
                "limit 默认 20，范围为 1 到 50，返回最近匹配记录。"
            ),
            input_schema=history_schema,
            handler=query_experiment_history,
            operation=ToolOperation.READ_ONLY,
            requires_intent_scope=True,
        ),
        ToolDefinition(
            name="get_experience_version",
            description=(
                "读取当前经验及其版本，可选 experience_id 必须匹配当前经验；"
                "不存在时返回 NOT_FOUND。"
            ),
            input_schema=experience_schema,
            handler=get_experience_version,
            operation=ToolOperation.READ_ONLY,
            requires_intent_scope=True,
        ),
    ])


def _validate_schema(value: Any, schema: Dict[str, Any], path: str) -> None:
    if "const" in schema and value != schema["const"]:
        raise ToolValidationError(
            f"{path} 必须等于 {schema['const']!r}，实际为 {value!r}"
        )

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} 必须是对象")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolValidationError(f"{path} 缺少必填字段: {missing}")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolValidationError(
                    f"{path} 包含未知字段: {sorted(unknown)}"
                )
        for name, child in value.items():
            if name in properties:
                _validate_schema(child, properties[name], f"{path}.{name}")
        return

    if expected_type == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} 必须是字符串")
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(
                f"{path} 的长度不能小于 {schema['minLength']}"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(
                f"{path} 的长度不能大于 {schema['maxLength']}"
            )
        return

    if expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolValidationError(f"{path} 必须是整数")
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolValidationError(f"{path} 必须是数字")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path} 必须是布尔值")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} 必须是数组")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], f"{path}[{index}]")
        return
    elif expected_type is not None:
        raise ToolValidationError(f"{path} 使用了不支持的 Schema 类型: {expected_type}")

    if "minimum" in schema and value < schema["minimum"]:
        raise ToolValidationError(f"{path} 不能小于 {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise ToolValidationError(f"{path} 不能大于 {schema['maximum']}")
