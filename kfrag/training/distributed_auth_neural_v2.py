"""Protocol-v2 packet generation and decoded-fragment evaluation utilities."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from kfrag.crypto.token import ProvenanceToken
from kfrag.protocols.distributed_auth_v2 import AuthFragment, JointFragmentCode


ACTIVE_BITS = 20
INDEX_SLICE = slice(0, 4)
RS_SLICE = slice(4, 12)
AUTH_SLICE = slice(12, 20)


def deterministic_scientific_key(seed: int) -> bytes:
    """Derive an in-memory test key. The return value must never be serialized."""
    return hashlib.sha256(f"kfrag-stage-d-v2-feasibility:{int(seed)}".encode()).digest()


def _uint_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def decoded_fragment(bits: torch.Tensor) -> AuthFragment:
    values = bits.detach().ge(.5).to(torch.uint8).cpu().tolist()
    index = sum(values[i] << (3 - i) for i in range(4))
    symbol = sum(values[4 + i] << (7 - i) for i in range(8))
    share = sum(values[12 + i] << (7 - i) for i in range(8))
    return AuthFragment(index, symbol, share)


@dataclass(frozen=True)
class IssuedGrid:
    token: ProvenanceToken
    source_id: bytes


def fresh_distributed_packet_batch(count: int, key: bytes, generator: torch.Generator):
    """Issue real joint-code packets; auth shares are never independent targets."""
    protocol = JointFragmentCode()
    grids, metadata = [], []
    for _ in range(count):
        issuer = int(torch.randint(0, 1 << 24, (1,), generator=generator))
        hi = int(torch.randint(0, 1 << 32, (1,), generator=generator))
        lo = int(torch.randint(0, 1 << 32, (1,), generator=generator))
        token = ProvenanceToken(issuer, (hi << 32) | lo, 2)
        source_id = bytes(torch.randint(0, 256, (8,), generator=generator, dtype=torch.uint8).tolist())
        fragments = protocol.issue(token, source_id, key)
        rows = [_uint_bits(f.index, 4) + _uint_bits(f.symbol, 8) + _uint_bits(f.share, 8) for f in fragments]
        grids.append(torch.tensor(rows, dtype=torch.float32).reshape(4, 4, ACTIVE_BITS))
        metadata.append(IssuedGrid(token, source_id))
    return torch.stack(grids), metadata


def fragments_from_logits(logits: torch.Tensor) -> list[list[AuthFragment]]:
    hard = logits[..., :ACTIVE_BITS].ge(0).float().reshape(len(logits), 16, ACTIVE_BITS)
    return [[decoded_fragment(row) for row in grid] for grid in hard]


def evaluate_protocol_controls(logits: torch.Tensor, metadata: list[IssuedGrid], key: bytes,
                               second_logits: torch.Tensor | None = None,
                               second_metadata: list[IssuedGrid] | None = None) -> dict:
    """Run protocol recovery solely on thresholded blind-decoder outputs."""
    protocol = JointFragmentCode()
    decoded = fragments_from_logits(logits)
    second = fragments_from_logits(second_logits) if second_logits is not None else decoded[1:] + decoded[:1]
    second_meta = second_metadata if second_metadata is not None else metadata[1:] + metadata[:1]
    counters = {name: 0 for name in (
        "token", "authenticator", "identity", "missing4", "auth8", "shuffled",
        "duplicate_rejected", "corrupt1_rejected", "corrupt2_rejected", "mixed_rejected",
        "insufficient_rejected")}
    state_counts = {"valid": 0, "missing": 0, "manipulated": 0, "unverified": 0}
    for n, (fragments, issued) in enumerate(zip(decoded, metadata)):
        result = protocol.recover_and_verify(fragments, issued.source_id, key)
        counters["token"] += int(result["token"] == issued.token)
        counters["identity"] += int(result["status"] == "valid")
        for state in result["states"].values(): state_counts[state] = state_counts.get(state, 0) + 1
        try:
            recovered = protocol.reconstruct_authenticator(fragments)
            expected = protocol._shares(issued.token, issued.source_id, key)[:8]
            counters["authenticator"] += int(recovered == expected)
        except ValueError: pass
        missing4 = fragments[:12]
        counters["missing4"] += int(protocol.recover_and_verify(missing4, issued.source_id, key)["status"] == "valid")
        try:
            recovered8 = protocol.reconstruct_authenticator(fragments[:8])
            counters["auth8"] += int(recovered8 == protocol._shares(issued.token, issued.source_id, key)[:8])
        except ValueError: pass
        shuffled = list(reversed(fragments))
        counters["shuffled"] += int(protocol.recover_and_verify(shuffled, issued.source_id, key)["status"] == "valid")
        duplicate = fragments[:-1] + [fragments[0]]
        counters["duplicate_rejected"] += int(protocol.recover_and_verify(duplicate, issued.source_id, key)["status"] != "valid")
        one = list(fragments); f = one[0]; one[0] = AuthFragment(f.index, f.symbol ^ 1, f.share)
        counters["corrupt1_rejected"] += int(protocol.recover_and_verify(one, issued.source_id, key)["status"] != "valid")
        two = list(one); f = two[1]; two[1] = AuthFragment(f.index, f.symbol, f.share ^ 1)
        counters["corrupt2_rejected"] += int(protocol.recover_and_verify(two, issued.source_id, key)["status"] != "valid")
        other = second[n % len(second)]
        mixed = fragments[:8] + other[8:]
        counters["mixed_rejected"] += int(protocol.recover_and_verify(mixed, issued.source_id, key)["status"] != "valid")
        counters["insufficient_rejected"] += int(protocol.recover_and_verify(fragments[:7], issued.source_id, key)["status"] == "insufficient")
    total = max(1, len(decoded))
    return {
        "token_reconstruction_success": counters["token"] / total,
        "authenticator_reconstruction_success": counters["authenticator"] / total,
        "authenticated_identity_acceptance": counters["identity"] / total,
        "twelve_valid_four_missing_acceptance": counters["missing4"] / total,
        "eight_share_authenticator_reconstruction": counters["auth8"] / total,
        "shuffled_region_acceptance": counters["shuffled"] / total,
        "duplicate_index_rejection": counters["duplicate_rejected"] / total,
        "one_corrupted_packet_rejection": counters["corrupt1_rejected"] / total,
        "two_corrupted_packet_rejection": counters["corrupt2_rejected"] / total,
        "mixed_identity_rejection": counters["mixed_rejected"] / total,
        "insufficient_evidence_rejection": counters["insufficient_rejected"] / total,
        "fragment_state_counts": state_counts,
        "protocol_input": "thresholded_blind_decoder_logits",
    }

