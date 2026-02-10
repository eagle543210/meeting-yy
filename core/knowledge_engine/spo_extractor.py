# 临时简化实现
from typing import List, Tuple

class SPOExtractor:
    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """示例规则实现"""
        if "总部位于" in text:
            parts = text.split("总部位于")
            return [(parts[0].strip(), "位置", parts[1].strip())]
        return []