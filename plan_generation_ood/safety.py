from typing import Protocol

from .recovery_schemas import (
    PermissionDecision,
    PermissionOutcome,
    RecoveryContext,
    ToolCall,
)
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
