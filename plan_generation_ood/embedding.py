from typing import List, Protocol


class EmbeddingPort(Protocol):
    async def embed(
        self,
        texts: List[str],
        model: str = "bge-m3",
    ) -> List[List[float]]:
        ...
