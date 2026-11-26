import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .schemas import DictSerializable


DEFAULT_SKILL_NAMES = frozenset({
    "experience_revision",
    "diagnosis_summary",
    "initial_experience",
})
DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parent / "skills"


class SkillLoadError(ValueError):
    pass


@dataclass(frozen=True)
class SkillMetadata(DictSerializable):
    name: str
    version: str
    description: str


@dataclass(frozen=True)
class SkillDocument(DictSerializable):
    metadata: SkillMetadata
    content: str
    content_hash: str


class SkillLoader:
    def __init__(
        self,
        root: Optional[str | Path] = None,
        allowed_names: Optional[Iterable[str]] = None,
        max_bytes: int = 65536,
    ):
        self.root = Path(root) if root is not None else DEFAULT_SKILLS_ROOT
        self.root = self.root.resolve()
        names = DEFAULT_SKILL_NAMES if allowed_names is None else frozenset(allowed_names)
        if not names:
            raise ValueError("Skill 白名单不能为空")
        for name in names:
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"非法 Skill 白名单名称: {name}")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes 必须是正整数")
        self.allowed_names = frozenset(names)
        self.max_bytes = max_bytes

    def list_skills(self) -> List[SkillMetadata]:
        return [
            self.load(name).metadata
            for name in sorted(self.allowed_names)
        ]

    def load(self, name: str) -> SkillDocument:
        path = self._path_for(name)
        try:
            if path.stat().st_size > self.max_bytes:
                raise SkillLoadError(f"Skill 文件超过大小上限: {name}")
            text = path.read_text(encoding="utf-8")
        except SkillLoadError:
            raise
        except (OSError, UnicodeDecodeError) as error:
            raise SkillLoadError(f"无法读取 Skill: {name}") from error
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        metadata, content = self._parse(name, normalized)
        return SkillDocument(
            metadata=metadata,
            content=content,
            content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        )

    def _path_for(self, name: str) -> Path:
        if not isinstance(name, str) or name not in self.allowed_names:
            raise SkillLoadError(f"未知或不在白名单中的 Skill: {name}")
        skill_directory = self.root / name
        path = skill_directory / "SKILL.md"
        if skill_directory.is_symlink() or path.is_symlink():
            raise SkillLoadError(f"Skill 不允许使用符号链接: {name}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as error:
            raise SkillLoadError(f"Skill 路径越界或不存在: {name}") from error
        if not resolved.is_file():
            raise SkillLoadError(f"Skill 不是普通文件: {name}")
        return resolved

    def _parse(self, name: str, text: str) -> Tuple[SkillMetadata, str]:
        lines = text.split("\n")
        if not lines or lines[0] != "---":
            raise SkillLoadError(f"Skill 缺少 frontmatter: {name}")
        try:
            closing = lines.index("---", 1)
        except ValueError as error:
            raise SkillLoadError(f"Skill frontmatter 未闭合: {name}") from error
        values: Dict[str, str] = {}
        for line in lines[1:closing]:
            if not line.strip():
                continue
            if ":" not in line:
                raise SkillLoadError(f"Skill 元数据必须是 key: value: {name}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key not in {"name", "version", "description"} or not value:
                raise SkillLoadError(f"Skill 元数据字段或值非法: {name}")
            if key in values:
                raise SkillLoadError(f"Skill 元数据字段重复: {key}")
            values[key] = value
        if set(values) != {"name", "version", "description"}:
            raise SkillLoadError(f"Skill 元数据必须完整提供 name/version/description: {name}")
        if values["name"] != name:
            raise SkillLoadError(f"Skill 元数据名称与目录不一致: {name}")
        content = "\n".join(lines[closing + 1:]).strip()
        if not content:
            raise SkillLoadError(f"Skill 正文不能为空: {name}")
        return SkillMetadata(
            name=name,
            version=values["version"],
            description=values["description"],
        ), content
