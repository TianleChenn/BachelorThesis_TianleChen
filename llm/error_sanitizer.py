from __future__ import annotations
import re
def sanitize_exception(exc:Exception)->str:
    message=f"{type(exc).__name__}: {exc}"
    message=re.sub(r"sk-[A-Za-z0-9_-]+","[REDACTED_API_KEY]",message)
    message=re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]+",r"\1[REDACTED]",message)
    message=re.sub(r"(?i)(api[_ -]?key\s*[=:]\s*)\S+",r"\1[REDACTED]",message)
    message=re.sub(r"(?s)(SYSTEM:|hidden prompt:).*","[REDACTED_PROMPT]",message)
    return message[:500]
