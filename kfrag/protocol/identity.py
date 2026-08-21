"""Opaque registered 96-bit identities."""
from __future__ import annotations
from dataclasses import dataclass
import secrets

@dataclass(frozen=True)
class RegisteredIdentity:
    value: bytes
    def __post_init__(self) -> None:
        if not isinstance(self.value, bytes) or len(self.value) != 12:
            raise ValueError("registered identity must be exactly 96 bits")
    @classmethod
    def generate(cls) -> "RegisteredIdentity":
        return cls(secrets.token_bytes(12))
    @classmethod
    def from_hex(cls, value: str) -> "RegisteredIdentity":
        try: raw = bytes.fromhex(value)
        except ValueError as exc: raise ValueError("invalid identity hex") from exc
        return cls(raw)
    def hex(self) -> str: return self.value.hex()
