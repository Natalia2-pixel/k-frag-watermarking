"""Domain-separated fragment authentication; keys are runtime-only values."""
from __future__ import annotations
import hashlib, hmac, os

DOMAIN = b"K-FRAG\x00FRAGMENT-AUTH\x00"
PROTOCOL_VERSION = 1

def runtime_key(env: str = "KFRAG_HMAC_KEY") -> bytes:
    value = os.environ.get(env)
    if not value: raise RuntimeError(f"set {env} at runtime; secret keys are never stored in artifacts")
    return value.encode("utf-8")

def authentication_message(namespace: bytes, index: int, symbol: int, version: int = PROTOCOL_VERSION) -> bytes:
    if not isinstance(namespace, bytes) or not namespace: raise ValueError("asset_namespace must be non-empty bytes")
    if len(namespace) > 65535: raise ValueError("asset_namespace is too long")
    if not 0 <= index < 16 or not 0 <= symbol < 256 or not 0 <= version < 256: raise ValueError("invalid packet field")
    return DOMAIN + bytes((version,)) + len(namespace).to_bytes(2, "big") + namespace + bytes((index, symbol))

def tag(key: bytes, namespace: bytes, index: int, symbol: int, tag_bits: int = 32, version: int = PROTOCOL_VERSION) -> bytes:
    if not isinstance(key, bytes) or not key: raise ValueError("key must be non-empty bytes")
    if tag_bits % 8 or not 8 <= tag_bits <= 256: raise ValueError("tag_bits must be byte-aligned in [8,256]")
    return hmac.new(key, authentication_message(namespace,index,symbol,version), hashlib.sha256).digest()[:tag_bits//8]

def verify(key: bytes, namespace: bytes, index: int, symbol: int, candidate: bytes, tag_bits: int = 32, version: int = PROTOCOL_VERSION) -> bool:
    if not isinstance(candidate, bytes) or len(candidate) != tag_bits // 8: return False
    return hmac.compare_digest(tag(key,namespace,index,symbol,tag_bits,version), candidate)
