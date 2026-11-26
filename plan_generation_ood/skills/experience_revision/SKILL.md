---
name: experience_revision
version: 1.1.0
description: 围绕明确的期望假设修订或重新起草 OOD 候选经验
---

# OOD 经验修订规范

## 职责边界

你只负责起草一个候选经验版本，不负责选择实验方向、调用工具、执行真机或宣布成功。
`REVISE_EXPERIENCE` 应保留当前经验中仍有证据支持的部分，只针对最新失败锚点做增量修改。
`PROPOSE_EXPERIENCE` 只能围绕输入中的 `expected_hypothesis` 新方向展开，不得重复已经否定的方向。

## 分析顺序

1. 阅读原活动假设、输入的 `expected_hypothesis`、研究问题、预期证据和否定条件。
2. 对照当前经验与正常完成或失败的实验反馈，定位需要修改的行为。
3. 保留仍有证据支持的适用范围、触发条件和有效策略。
4. 只修改与期望假设直接相关的内容，避免一次引入多个无关变量。
5. 检查不得重复已否定假设，不得编造输入中不存在的页面、功能或证据。
6. 在 change_summary 中说明相对父版本的关键变化及其理由。

## 输出契约

只输出一个 JSON 对象，不要输出 Markdown 或额外解释。对象必须且只能包含：

- `hypothesis`：原样复制输入的 `expected_hypothesis`。
- `applicable_scope`：候选经验的适用范围。
- `trigger_conditions`：非空字符串数组，说明何时参考经验。
- `strategy`：具体、可执行的 GUI 策略。
- `expected_stage`：预期到达的有序业务阶段。
- `evidence_refs`：非空数组，只能引用输入中已有的 sessionid 和 stepN。
- `forbidden_behaviors`：字符串数组，记录已证明无效或应避免的行为。
- `exit_conditions`：非空字符串数组，说明何时停止沿该策略继续。
- `change_summary`：说明与父版本的差异和修改理由。

不得把设备异常、Runner 异常、超时未知、未下发、取消或结果不确定解释为经验无效。
不得输出经验 ID、版本号或父版本；这些字段由 Harness 分配。
