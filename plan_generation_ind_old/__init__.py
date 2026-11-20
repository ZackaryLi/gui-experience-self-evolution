"""
Success Experience Generation Pipeline
从已成功轨迹中提取高效经验的Pipeline (多Plan版)

数据格式: list[dict]，每条dict是一个轨迹，intentName字段表示意图
一个pipeline运行处理一个intentName下的所有trajectory
生成多个Plan，每个针对不同的高效策略或冗余模式
"""
from .pipeline import PlanGenerationPipeline
from .confidence import ConfidenceConfig, Calibrator, NoopCalibrator

__all__ = [
    "PlanGenerationPipeline",
    "ConfidenceConfig",
    "Calibrator",
    "NoopCalibrator",
]
