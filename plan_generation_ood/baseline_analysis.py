import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._analysis_llm import StructuredAnalysisClient, StructuredAnalysisError
from .config import OODBaselineAnalysisConfig
from .progress import BusinessStageAnalyzer
from .schemas import (
    AttemptAnalysis,
    AttemptResult,
    AttemptStatus,
    BaselineAnalysisIssue,
    BaselinePlan,
    BaselineStatistics,
    BusinessStageRubric,
    DiagnosisReport,
    EvidenceRef,
    FailurePattern,
    content_hash,
)
from .skills import SkillLoader
from .storage import atomic_write_json
from .verdict import AttemptVerdictAnalyzer, result_sessionid, trajectory_steps


class BaselineAnalysisError(ValueError):
    pass


class BaselineAnalyzer:
    NORMAL_STATUSES = {
        AttemptStatus.FINISHED,
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
    }
    INFRASTRUCTURE_STATUSES = {
        AttemptStatus.DEVICE_ERROR,
        AttemptStatus.RUNNER_ERROR,
        AttemptStatus.TIMEOUT_UNKNOWN,
    }

    def __init__(
        self,
        llm_agent,
        model: str,
        config: OODBaselineAnalysisConfig | None = None,
        skill_loader: Optional[SkillLoader] = None,
    ):
        if skill_loader is not None and not isinstance(skill_loader, SkillLoader):
            raise ValueError("skill_loader 必须是 SkillLoader 或 None")
        self.config = config or OODBaselineAnalysisConfig()
        self.verdict_analyzer = AttemptVerdictAnalyzer(
            llm_agent,
            model,
            self.config,
        )
        self.progress_analyzer = BusinessStageAnalyzer(
            llm_agent,
            model,
            self.config,
        )
        self.summary_client = StructuredAnalysisClient(
            llm_agent,
            model,
            self.config,
        )
        self.skill_loader = skill_loader

    def analyze(
        self,
        plan: BaselinePlan,
        results: List[AttemptResult],
        stage_rubric: Optional[BusinessStageRubric] = None,
        output_path: str | Path | None = None,
    ) -> DiagnosisReport:
        return asyncio.run(self.aanalyze(
            plan=plan,
            results=results,
            stage_rubric=stage_rubric,
            output_path=output_path,
        ))

    async def aanalyze(
        self,
        plan: BaselinePlan,
        results: List[AttemptResult],
        stage_rubric: Optional[BusinessStageRubric] = None,
        output_path: str | Path | None = None,
    ) -> DiagnosisReport:
        ordered_results = self._validate_inputs(plan, results)
        if stage_rubric is not None:
            try:
                stage_rubric = self.progress_analyzer.validate_rubric(
                    stage_rubric,
                    plan.intent_id,
                )
            except (TypeError, ValueError) as error:
                raise BaselineAnalysisError(str(error)) from error

        issues: List[BaselineAnalysisIssue] = []
        analyses: List[AttemptAnalysis] = []
        result_by_key = {
            result.request.idempotency_key: result for result in ordered_results
        }
        expected_requests = plan.to_attempt_requests()
        received_keys = set(result_by_key)
        for request in expected_requests:
            if request.idempotency_key not in received_keys:
                issues.append(BaselineAnalysisIssue(
                    code="ATTEMPT_MISSING",
                    message="冻结计划中的 attempt 尚未回收",
                    attempt_key=request.idempotency_key,
                    instruction_id=request.instruction_id,
                ))

        for result in ordered_results:
            analysis = await self._analyze_attempt(result, issues)
            analyses.append(analysis)

        valid_results = [
            result_by_key[analysis.attempt_key]
            for analysis in analyses
            if analysis.included_in_strategy
        ]
        if stage_rubric is None and valid_results:
            try:
                stage_rubric = await self.progress_analyzer.abuild_rubric(
                    plan.intent_id,
                    valid_results,
                )
            except StructuredAnalysisError as error:
                issues.append(BaselineAnalysisIssue(
                    code="STAGE_RUBRIC_UNAVAILABLE",
                    message=str(error),
                ))

        progress_by_attempt = {}
        if stage_rubric is not None:
            for analysis in analyses:
                if not analysis.included_in_strategy or analysis.verdict is None:
                    continue
                result = result_by_key[analysis.attempt_key]
                try:
                    progress = await self.progress_analyzer.aanalyze(
                        result,
                        stage_rubric,
                        analysis.verdict,
                    )
                    progress_by_attempt[analysis.attempt_key] = progress
                except StructuredAnalysisError as error:
                    self._record_issue(
                        issues,
                        analysis,
                        "PROGRESS_UNAVAILABLE",
                        str(error),
                    )

        overall = self._statistics(
            planned_count=plan.total_attempts,
            analyses=analyses,
        )
        by_instruction = {}
        for sample in plan.samples:
            by_instruction[sample.instruction_id] = self._statistics(
                planned_count=len(sample.attempt_indices),
                analyses=[
                    analysis for analysis in analyses
                    if analysis.instruction_id == sample.instruction_id
                ],
            )
        patterns = self._failure_patterns(analyses, progress_by_attempt)
        evidence_quality = self._evidence_quality(
            plan,
            analyses,
            progress_by_attempt,
        )
        report = DiagnosisReport(
            intent_id=plan.intent_id,
            baseline_plan_id=plan.plan_id,
            attempt_results=list(ordered_results),
            progress_by_attempt=progress_by_attempt,
            failure_patterns=patterns,
            summary=self._deterministic_summary(overall, patterns),
            evidence_quality=evidence_quality,
            attempt_analyses=analyses,
            overall_statistics=overall,
            statistics_by_instruction=by_instruction,
            stage_rubric=stage_rubric,
            issues=issues,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._apply_summary_skill(report)
        if output_path is not None:
            atomic_write_json(output_path, report.to_dict())
        return report

    def _validate_inputs(
        self,
        plan: BaselinePlan,
        results: List[AttemptResult],
    ) -> List[AttemptResult]:
        if not isinstance(plan, BaselinePlan) or plan.status != "FROZEN":
            raise BaselineAnalysisError("plan 必须是状态为 FROZEN 的 BaselinePlan")
        if not isinstance(results, list) or any(
            not isinstance(result, AttemptResult) for result in results
        ):
            raise BaselineAnalysisError("results 必须是 AttemptResult 数组")
        if not results:
            raise BaselineAnalysisError("没有基线结果，调用方应保持 BASELINE_PENDING")

        expected = {}
        order = {}
        for index, request in enumerate(plan.to_attempt_requests()):
            identity = (request.instruction_id, request.attempt_index)
            if request.idempotency_key in expected or identity in order:
                raise BaselineAnalysisError("冻结计划包含重复 attempt 身份")
            expected[request.idempotency_key] = request
            order[identity] = index

        seen = set()
        for result in results:
            request = result.request
            key = request.idempotency_key
            identity = (request.instruction_id, request.attempt_index)
            if key in seen:
                raise BaselineAnalysisError(f"收到重复 attempt: {key}")
            seen.add(key)
            expected_request = expected.get(key)
            if expected_request is None or identity not in order:
                raise BaselineAnalysisError(f"收到冻结计划之外的 attempt: {key}")
            if (
                request.intent_id != plan.intent_id
                or request.plan_id != plan.plan_id
                or request.instruction_id != expected_request.instruction_id
                or request.instruction != expected_request.instruction
                or request.attempt_index != expected_request.attempt_index
            ):
                raise BaselineAnalysisError(f"attempt 请求与冻结计划不一致: {key}")
            if request.experience is not None:
                raise BaselineAnalysisError(f"无经验基线不能携带经验: {key}")
            if not isinstance(result.trajectory, dict):
                raise BaselineAnalysisError(f"attempt trajectory 必须是字典: {key}")
            if not isinstance(result.status, AttemptStatus):
                raise BaselineAnalysisError(f"attempt 执行状态非法: {key}")
        return sorted(
            results,
            key=lambda result: order[(
                result.request.instruction_id,
                result.request.attempt_index,
            )],
        )

    async def _analyze_attempt(
        self,
        result: AttemptResult,
        issues: List[BaselineAnalysisIssue],
    ) -> AttemptAnalysis:
        request = result.request
        if result.status in self.NORMAL_STATUSES:
            execution_class = "NORMAL"
        elif result.status is AttemptStatus.UNCERTAIN:
            execution_class = "UNCERTAIN"
        elif result.status in self.INFRASTRUCTURE_STATUSES:
            execution_class = "INFRASTRUCTURE_OR_UNKNOWN"
        else:
            execution_class = "NOT_EXECUTED"
        analysis = AttemptAnalysis(
            attempt_key=request.idempotency_key,
            instruction_id=request.instruction_id,
            attempt_index=request.attempt_index,
            sessionid=result_sessionid(result),
            execution_status=result.status,
            execution_class=execution_class,
        )
        if result.status not in self.NORMAL_STATUSES:
            return analysis

        if result.status in {AttemptStatus.COMPLETED, AttemptStatus.FAILED}:
            expected_flag = result.status is AttemptStatus.COMPLETED
            if result.task_completed is not expected_flag:
                analysis.conflict = True
                self._record_issue(
                    issues,
                    analysis,
                    "COMPLETION_FLAG_CONFLICT",
                    "原始业务状态与 task_completed 不一致，保留冲突并排除策略统计",
                )
        elif result.task_completed is not None:
            analysis.conflict = True
            self._record_issue(
                issues,
                analysis,
                "FINISHED_WITH_BUSINESS_FLAG",
                "业务中性的 finished 不应预设 task_completed，保留冲突并排除策略统计",
            )
        if not analysis.sessionid or not trajectory_steps(result.trajectory):
            self._record_issue(
                issues,
                analysis,
                "TRAJECTORY_INVALID",
                "正常 attempt 缺少 sessionid 或有效 stepN",
            )
            return analysis

        try:
            verdict = await self.verdict_analyzer.aanalyze(result)
            analysis.verdict = verdict
        except StructuredAnalysisError as error:
            self._record_issue(
                issues,
                analysis,
                "VERDICT_UNAVAILABLE",
                str(error),
            )
            return analysis

        if verdict.completion_status is AttemptStatus.UNCERTAIN:
            self._record_issue(
                issues,
                analysis,
                "VERDICT_UNCERTAIN",
                "独立 Verdict 证据不足，不参与策略统计",
            )
            return analysis
        if result.status in {AttemptStatus.COMPLETED, AttemptStatus.FAILED}:
            expected_status = (
                AttemptStatus.COMPLETED
                if verdict.completion_status is AttemptStatus.COMPLETED
                else AttemptStatus.FAILED
            )
            if result.status is not expected_status:
                analysis.conflict = True
                self._record_issue(
                    issues,
                    analysis,
                    "VERDICT_STATUS_CONFLICT",
                    "独立 Verdict 与原始业务状态冲突，不自动覆盖任何一方",
                )
        if (
            result.task_completed is not None
            and result.task_completed != verdict.task_completed
        ):
            analysis.conflict = True
            self._record_issue(
                issues,
                analysis,
                "VERDICT_FLAG_CONFLICT",
                "独立 Verdict 与原始 task_completed 冲突",
            )
        if verdict.confidence < self.config.min_verdict_confidence:
            self._record_issue(
                issues,
                analysis,
                "VERDICT_LOW_CONFIDENCE",
                "独立 Verdict 置信度低于配置阈值",
            )
        if not analysis.conflict and not analysis.issues:
            analysis.included_in_strategy = True
        return analysis

    @staticmethod
    def _record_issue(
        issues: List[BaselineAnalysisIssue],
        analysis: AttemptAnalysis,
        code: str,
        message: str,
    ) -> None:
        if code not in analysis.issues:
            analysis.issues.append(code)
        issues.append(BaselineAnalysisIssue(
            code=code,
            message=message,
            attempt_key=analysis.attempt_key,
            instruction_id=analysis.instruction_id,
        ))

    @staticmethod
    def _statistics(
        planned_count: int,
        analyses: List[AttemptAnalysis],
    ) -> BaselineStatistics:
        stats = BaselineStatistics(planned_count=planned_count)
        for analysis in analyses:
            stats.received_count += 1
            field_name = f"{analysis.execution_status.value}_count"
            if hasattr(stats, field_name):
                setattr(stats, field_name, getattr(stats, field_name) + 1)
            if analysis.conflict:
                stats.conflict_count += 1
            if analysis.included_in_strategy and analysis.verdict is not None:
                stats.valid_attempt_keys.append(analysis.attempt_key)
                if analysis.verdict.completion_status is AttemptStatus.COMPLETED:
                    stats.valid_completed_count += 1
                else:
                    stats.valid_failed_count += 1
            else:
                stats.excluded_attempt_keys.append(analysis.attempt_key)
        return stats

    @staticmethod
    def _failure_patterns(
        analyses: List[AttemptAnalysis],
        progress_by_attempt,
    ) -> List[FailurePattern]:
        groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for analysis in analyses:
            verdict = analysis.verdict
            if (
                not analysis.included_in_strategy
                or verdict is None
                or verdict.completion_status is not AttemptStatus.FAILED
            ):
                continue
            progress = progress_by_attempt.get(analysis.attempt_key)
            stage_name = progress.stage_name if progress is not None else "UNKNOWN"
            key = (
                verdict.primary_failure_parent_category,
                verdict.primary_failure_subcategory,
                stage_name,
                verdict.blocker_type,
            )
            group = groups.setdefault(key, {
                "attempt_keys": [],
                "instruction_ids": set(),
                "evidence_refs": [],
            })
            group["attempt_keys"].append(analysis.attempt_key)
            group["instruction_ids"].add(analysis.instruction_id)
            group["evidence_refs"].append(EvidenceRef(
                sessionid=verdict.sessionid,
                step_ids=list(verdict.key_evidence_steps),
            ))

        patterns = []
        for key, group in sorted(groups.items()):
            parent, subcategory, stage_name, blocker_type = key
            pattern_id = content_hash({
                "parent": parent,
                "subcategory": subcategory,
                "stage_name": stage_name,
                "blocker_type": blocker_type,
            })[:20]
            patterns.append(FailurePattern(
                pattern_id=pattern_id,
                name=f"{subcategory} @ {stage_name}",
                description=(
                    f"{len(group['attempt_keys'])} 个有效失败 attempt 在 {stage_name} 阶段 "
                    f"呈现 {subcategory} 分类，覆盖 {len(group['instruction_ids'])} 条指令；"
                    "该分组不自动代表任务不可解或具有修复价值。"
                ),
                evidence_refs=group["evidence_refs"],
                blocker_type=blocker_type,
            ))
        return patterns

    @staticmethod
    def _evidence_quality(
        plan: BaselinePlan,
        analyses: List[AttemptAnalysis],
        progress_by_attempt,
    ) -> str:
        valid_count = sum(analysis.included_in_strategy for analysis in analyses)
        if any(analysis.conflict for analysis in analyses):
            return "CONFLICT"
        if valid_count == 0:
            return "INSUFFICIENT"
        if (
            len(analyses) == plan.total_attempts
            and valid_count == plan.total_attempts
            and len(progress_by_attempt) == valid_count
        ):
            return "SUFFICIENT"
        return "PARTIAL"

    @staticmethod
    def _deterministic_summary(
        stats: BaselineStatistics,
        patterns: List[FailurePattern],
    ) -> str:
        infrastructure = (
            stats.device_error_count
            + stats.runner_error_count
            + stats.timeout_unknown_count
        )
        summary = (
            f"计划 {stats.planned_count} 个 attempt，回收 {stats.received_count} 个；"
            f"有效完成 {stats.valid_completed_count} 个，有效失败 {stats.valid_failed_count} 个，"
            f"不确定 {stats.uncertain_count} 个，基础设施异常或超时未知 {infrastructure} 个。"
        )
        if patterns:
            summary += " 保守失败分组：" + "；".join(pattern.name for pattern in patterns) + "。"
        else:
            summary += " 当前没有可用于策略失败分组的明确证据。"
        return summary

    async def _apply_summary_skill(self, report: DiagnosisReport) -> None:
        if self.skill_loader is None:
            return
        try:
            document = self.skill_loader.load("diagnosis_summary")
            payload = {
                "intent_id": report.intent_id,
                "baseline_plan_id": report.baseline_plan_id,
                "overall_statistics": report.overall_statistics.to_dict(),
                "statistics_by_instruction": {
                    key: value.to_dict()
                    for key, value in report.statistics_by_instruction.items()
                },
                "progress_by_attempt": {
                    key: value.to_dict()
                    for key, value in report.progress_by_attempt.items()
                },
                "failure_patterns": [item.to_dict() for item in report.failure_patterns],
                "issues": [item.to_dict() for item in report.issues],
                "evidence_quality": report.evidence_quality,
            }

            def validate(raw: Any) -> str:
                if not isinstance(raw, dict) or set(raw) != {"summary"}:
                    raise ValueError("摘要只能包含 summary 字段")
                summary = raw["summary"]
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("summary 必须是非空字符串")
                return summary.strip()

            summary, _ = await self.summary_client.request(
                "你是固定 OOD 诊断摘要器，只能总结输入事实。\n\n" + document.content,
                payload,
                validate,
            )
            report.summary = summary
            report.summary_skill_name = document.metadata.name
            report.summary_skill_version = document.metadata.version
            report.summary_skill_content_hash = document.content_hash
        except (TypeError, ValueError) as error:
            report.issues.append(BaselineAnalysisIssue(
                code="SUMMARY_UNAVAILABLE",
                message=f"自然语言摘要不可用，保留确定性摘要：{type(error).__name__}",
            ))
