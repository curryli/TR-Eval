"""Stable identifiers for source samples without upstream IDs."""

import hashlib
import json


def stable_id(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
