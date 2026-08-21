"""Oracle-channel protocol API.

This module validates protocol and provenance behavior independently of the
experimental learned image carrier.  The oracle stores one decoded packet per
logical image region; it does not claim that pixels can yet carry those bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from kfrag.evidence import Candidate, protocol_evidence_map, verify_candidates
from .identity import RegisteredIdentity
from .packet import FragmentPacket, create_fragments
from .reconstruction import reconstruct_identity


@dataclass(frozen=True)
class Registration:
    identity: RegisteredIdentity
    asset_namespace: bytes

    def public_metadata(self) -> dict[str, str]:
        return {"identity": self.identity.hex(), "asset_namespace": self.asset_namespace.hex()}


class IdentityRegistry:
    """Minimal verifier-side registry; authentication secrets are never stored."""

    def __init__(self) -> None:
        self._registrations: dict[bytes, Registration] = {}

    def register(self, asset_namespace: bytes, identity: RegisteredIdentity | None = None) -> Registration:
        if not isinstance(asset_namespace, bytes) or not asset_namespace:
            raise ValueError("asset_namespace must be non-empty bytes")
        if asset_namespace in self._registrations:
            raise ValueError("asset_namespace is already registered")
        registration = Registration(identity or RegisteredIdentity.generate(), asset_namespace)
        self._registrations[asset_namespace] = registration
        return registration

    def resolve(self, asset_namespace: bytes) -> Registration:
        try: return self._registrations[asset_namespace]
        except KeyError as exc: raise ValueError("unregistered asset namespace") from exc


@dataclass(frozen=True)
class OracleEmbeddedImage:
    """Questioned image plus simulated regional-symbol observations."""
    image: Any
    asset_namespace: bytes
    regional_packets: tuple[FragmentPacket | None, ...]

    def __post_init__(self) -> None:
        if len(self.regional_packets) != 16:
            raise ValueError("oracle channel requires exactly 16 regional observations")


@dataclass(frozen=True)
class VerificationResult:
    status: str
    identity: RegisteredIdentity | None
    evidence_map: list[list[str]]
    valid_indices: tuple[int, ...]
    invalid_indices: tuple[int, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "identity": None if self.identity is None else self.identity.hex(),
                "evidence_map": self.evidence_map, "valid_indices": list(self.valid_indices),
                "invalid_indices": list(self.invalid_indices), "errors": list(self.errors),
                "validation_track": "oracle_channel_protocol_validation"}


def embed(image: Any, registration: Registration, key: bytes) -> OracleEmbeddedImage:
    """Simulate embedding without requiring later access to the original image."""
    packets = create_fragments(registration.identity, key, registration.asset_namespace)
    return OracleEmbeddedImage(image, registration.asset_namespace, packets)


def verify_image(questioned_image: OracleEmbeddedImage, registry: IdentityRegistry, key: bytes) -> VerificationResult:
    """Blind protocol verification: no original, expected payload, or crop coordinates."""
    registration = registry.resolve(questioned_image.asset_namespace)
    candidates = [Candidate(packet, ((i % 4 + .5) / 4, (i // 4 + .5) / 4), 1.0,
                            observed=packet is not None, decodable=packet is not None)
                  for i, packet in enumerate(questioned_image.regional_packets)]
    accepted, rejected, conflicts = verify_candidates(candidates, key, registration.asset_namespace)
    raw_evidence = protocol_evidence_map(candidates, accepted, rejected, conflicts)
    labels = {"valid_authenticated": "valid", "missing_or_unobserved": "missing",
              "invalid_authentication": "invalid/manipulated",
              "duplicate_or_conflicting": "invalid/manipulated",
              "undecodable": "unavailable/uncertain"}
    evidence = [[labels[state] for state in row] for row in raw_evidence]
    invalid = tuple(sorted({c.packet.index for c, _ in rejected if c.packet is not None} | set(conflicts)))
    errors: list[str] = []
    identity = None
    if conflicts: errors.append("duplicate_index")
    try:
        identity = reconstruct_identity((c.packet for c in accepted.values()), key,
                                        registration.asset_namespace)
        if identity != registration.identity:
            errors.append("registered_identity_mismatch"); identity = None
    except ValueError as exc: errors.append(str(exc))
    status = "valid" if identity is not None and not invalid else "invalid"
    return VerificationResult(status, identity, evidence, tuple(sorted(accepted)), invalid, tuple(errors))
