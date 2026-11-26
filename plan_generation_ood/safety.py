from typing import Protocol

from .recovery_schemas import (
    PermissionDecision,
    PermissionOutcome,
    RecoveryContext,
    ToolCall,
)
from .schemas import RecoveryState
from .tools import ToolDefinition, ToolOperation


class PermissionPolicy(Protocol):
    def evaluate(
        self,
        context: RecoveryContext,
        definition: ToolDefinition,
        call: ToolCall,
    ) -> PermissionDecision:
        ...


class DefaultPermissionPolicy:
    def evaluate(
        self,
        context: RecoveryContext,
        definition: ToolDefinition,
        call: ToolCall,
    ) -> PermissionDecision:
        if definition.operation is ToolOperation.SENSITIVE:
            return self._decision(
                PermissionOutcome.DENY,
                "SENSITIVE_OPERATION",
                "敏感终态操作默认禁止，不能由 Agent 或人工参数覆盖。",
                context,
                definition,
            )

        if definition.operation is ToolOperation.UNKNOWN:
            return self._decision(
                PermissionOutcome.MANUAL_REVIEW,
                "UNKNOWN_TOOL_OPERATION",
                "工具未声明操作类型，默认不放行。",
                context,
                definition,
            )

        if definition.operation is ToolOperation.EXPERIMENT:
            return self._decision(
                PermissionOutcome.MANUAL_REVIEW,
                "EXPERIMENT_NOT_AUTHORIZED",
                "当前未配置实验执行授权，不能提交或控制实验。",
                context,
                definition,
            )

        if definition.operation is not ToolOperation.READ_ONLY:
            return self._decision(
                PermissionOutcome.MANUAL_REVIEW,
                "UNSUPPORTED_TOOL_OPERATION",
                "无法识别工具操作类型，默认不放行。",
                context,
                definition,
            )

        if definition.requires_intent_scope:
            requested_intent = call.input.get("intent_id")
            if not isinstance(requested_intent, str) or not requested_intent.strip():
                return self._decision(
                    PermissionOutcome.DENY,
                    "INTENT_SCOPE_REQUIRED",
                    "该只读工具必须显式提供当前 intent_id。",
                    context,
                    definition,
                )
            if requested_intent != context.intent_id:
                return self._decision(
                    PermissionOutcome.DENY,
                    "INTENT_SCOPE_VIOLATION",
                    "只读工具不能访问其他 intent 的数据。",
                    context,
                    definition,
                    {"requested_intent_id": requested_intent},
                )

        return self._decision(
            PermissionOutcome.ALLOW,
            "READ_ONLY_WITHIN_SCOPE",
            "只读工具及其数据范围符合当前权限策略。",
            context,
            definition,
        )

    @staticmethod
    def _decision(
        outcome: PermissionOutcome,
        reason_code: str,
        reason: str,
        context: RecoveryContext,
        definition: ToolDefinition,
        details=None,
    ) -> PermissionDecision:
        return PermissionDecision(
            outcome=outcome,
            reason_code=reason_code,
            reason=reason,
            tool=definition.name,
            intent_id=context.intent_id,
            details={
                "operation": definition.operation.value,
                **(details or {}),
            },
        )


class ExecutionPermissionPolicy:
    def __init__(self, execution_authorized: bool = False):
        self.execution_authorized = execution_authorized
        self.read_only = DefaultPermissionPolicy()

    def evaluate(
        self,
        context: RecoveryContext,
        definition: ToolDefinition,
        call: ToolCall,
    ) -> PermissionDecision:
        if definition.operation is not ToolOperation.EXPERIMENT:
            return self.read_only.evaluate(context, definition, call)
        if not self.execution_authorized:
            return DefaultPermissionPolicy._decision(
                PermissionOutcome.MANUAL_REVIEW,
                "EXPERIMENT_NOT_AUTHORIZED",
                "当前 Harness 未获得真机实验授权。",
                context,
                definition,
            )
        if context.state is not RecoveryState.EXPERIENCE_EXPLORING:
            return DefaultPermissionPolicy._decision(
                PermissionOutcome.DENY,
                "EXPERIMENT_STATE_DENIED",
                "只有 EXPERIENCE_EXPLORING 可以提交或取消探索实验。",
                context,
                definition,
            )
        if call.input.get("intent_id") != context.intent_id:
            return DefaultPermissionPolicy._decision(
                PermissionOutcome.DENY,
                "INTENT_SCOPE_VIOLATION",
                "实验工具不能访问其他 intent。",
                context,
                definition,
            )
        return DefaultPermissionPolicy._decision(
            PermissionOutcome.ALLOW,
            "EXPERIMENT_WITHIN_SCOPE",
            "实验工具已获得执行授权且 intent 范围一致；执行器仍将校验预算和计划。",
            context,
            definition,
        )
