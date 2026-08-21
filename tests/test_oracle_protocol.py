from dataclasses import replace
from kfrag.protocol import (IdentityRegistry, OracleEmbeddedImage, RegisteredIdentity,
                            embed, verify_image)
from kfrag.protocol.packet import FragmentPacket

KEY=b"oracle-test-key"
def fixture():
    registry=IdentityRegistry(); reg=registry.register(b"asset-a",RegisteredIdentity(bytes(range(12))))
    return registry,reg,embed(object(),reg,KEY)

def test_wire_packet_is_44_bits_with_zero_padding_nibble():
    _,_,image=fixture(); raw=image.regional_packets[0].to_bytes()
    assert len(raw)==6 and raw[-1]&15==0
    assert FragmentPacket.from_bytes(raw)==image.regional_packets[0]

def test_any_twelve_and_spatial_missing_evidence():
    registry,reg,image=fixture(); packets=list(image.regional_packets)
    for i in (0,3,8,15): packets[i]=None
    result=verify_image(replace(image,regional_packets=tuple(packets)),registry,KEY)
    assert result.status=="valid" and result.identity==reg.identity
    assert sum(x=="missing" for row in result.evidence_map for x in row)==4

def test_wrong_key_modified_symbol_tag_duplicate_and_cross_image_rejected():
    registry,_,image=fixture()
    assert verify_image(image,registry,b"wrong").status=="invalid"
    for field,value in (("symbol",image.regional_packets[0].symbol^1),("authentication_tag",b"\0"*4)):
        packets=list(image.regional_packets); packets[0]=replace(packets[0],**{field:value})
        assert verify_image(replace(image,regional_packets=tuple(packets)),registry,KEY).status=="invalid"
    packets=list(image.regional_packets); packets[1]=packets[0]
    assert verify_image(replace(image,regional_packets=tuple(packets)),registry,KEY).status=="invalid"
    other=registry.register(b"asset-b",RegisteredIdentity(b"x"*12)); substituted=embed(None,other,KEY)
    packets=list(image.regional_packets); packets[0]=substituted.regional_packets[0]
    assert verify_image(replace(image,regional_packets=tuple(packets)),registry,KEY).status=="invalid"

def test_fewer_than_twelve_is_invalid_and_api_is_blind():
    registry,_,image=fixture(); packets=tuple(list(image.regional_packets[:11])+[None]*5)
    result=verify_image(replace(image,regional_packets=packets),registry,KEY)
    assert result.status=="invalid" and "fewer than 12" in " ".join(result.errors)
