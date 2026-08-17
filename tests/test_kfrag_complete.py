import inspect, itertools
import pytest, torch
from kfrag.protocol.identity import RegisteredIdentity
from kfrag.protocol.packet import FragmentPacket,create_fragments
from kfrag.protocol.reconstruction import reconstruct_identity
from kfrag.protocol.authentication import verify
from kfrag.models import KFragSystem,BlindPacketDecoder
from kfrag.preprocessing import BlindNormalizer
from kfrag.evidence import Candidate,verify_candidates,protocol_evidence_map
from kfrag.data.manifests import assert_disjoint,manifest_hash

def test_exact_packet_serialization_and_hmac_binding():
    identity=RegisteredIdentity(bytes(range(12))); packets=create_fragments(identity,b"runtime-test",b"asset-A")
    for packet in packets:
        assert FragmentPacket.from_bits(packet.to_bits())==packet
        assert verify(b"runtime-test",b"asset-A",packet.index,packet.symbol,packet.authentication_tag)
        assert not verify(b"wrong",b"asset-A",packet.index,packet.symbol,packet.authentication_tag)
        assert not verify(b"runtime-test",b"asset-B",packet.index,packet.symbol,packet.authentication_tag)

def test_arbitrary_threshold_reconstruction_and_below_threshold_failure():
    identity=RegisteredIdentity(bytes(range(12))); packets=create_fragments(identity,b"runtime-test",b"asset")
    for missing in range(4): assert reconstruct_identity(packets[missing:missing+12],b"runtime-test",b"asset")==identity
    with pytest.raises(ValueError,match="fewer than 12"): reconstruct_identity(packets[:11],b"runtime-test",b"asset")
    with pytest.raises(ValueError,match="duplicate"): reconstruct_identity([packets[0],*packets[:12]],b"runtime-test",b"asset")

def test_decoder_and_preprocessing_are_blind():
    assert list(inspect.signature(BlindPacketDecoder.forward).parameters)==["self","questioned_image"]
    x=torch.rand(1,3,64,64)
    for mode in BlindNormalizer.MODES: assert BlindNormalizer(mode)(x).shape[0]==1

def test_payload_dependent_bounded_residual_and_candidate_evidence():
    model=KFragSystem(width=8,residual_alpha=.03); image=torch.rand(1,3,64,64); a=torch.zeros(1,44,4,4); b=torch.ones_like(a)
    ra=model.encode(image,a)["residual"]; rb=model.encode(image,b)["residual"]
    assert float(ra.abs().max().detach())<=.030001 and float((ra-rb).abs().mean().detach())>0
    packet=create_fragments(RegisteredIdentity(bytes(12)),b"runtime-test",b"asset")[0]; candidates=[Candidate(packet,(.1,.1),.9)]
    accepted,rejected,conflicts=verify_candidates(candidates,b"runtime-test",b"asset"); evidence=protocol_evidence_map(candidates,accepted,rejected,conflicts)
    assert evidence[0][0]=="valid_authenticated" and evidence[0][1]=="missing_or_unobserved"

def test_manifest_overlap_detection_and_stable_hash():
    manifest={"train":["a.jpg"],"validation":["b.jpg"],"test":["c.jpg"]}; assert_disjoint(manifest); assert manifest_hash(manifest)==manifest_hash(dict(manifest))
    with pytest.raises(ValueError,match="overlap"): assert_disjoint({"train":["a"],"validation":["a"],"test":[]})
