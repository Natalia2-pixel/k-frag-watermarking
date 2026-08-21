import hashlib,itertools,pytest
from kfrag.crypto.token import ProvenanceToken
from kfrag.protocols.distributed_auth_v2 import AuthFragment,IndependentMAC,DistributedGlobalMAC,JointFragmentCode,construction_comparison,monte_carlo

KEY=hashlib.sha256(b"test-key").digest();SOURCE=b"source-A";TOKEN=ProvenanceToken(7,123456789,2)
@pytest.mark.parametrize("bits",[8,12,16])
def test_independent_mac_probabilities_and_mutations(bits):
    construction=IndependentMAC(bits);fragments=construction.issue(TOKEN,SOURCE,KEY);assert construction.one_forge_probability()==2**-bits;assert construction.all_forges_probability(12)==2**(-12*bits)
    for field in ("symbol","share"):
        f=fragments[3];changed=AuthFragment(f.index,f.symbol^1,f.share,f.share_bits) if field=="symbol" else AuthFragment(f.index,f.symbol,f.share^1,f.share_bits);states=construction.verify(fragments[:3]+[changed]+fragments[4:],TOKEN,SOURCE,KEY);assert states[3]=="manipulated"
def test_unordered_erased_and_duplicate_states():
    c=IndependentMAC(8);fragments=c.issue(TOKEN,SOURCE,KEY);states=c.verify(list(reversed(fragments))[1:],TOKEN,SOURCE,KEY);assert sum(x=="missing" for x in states.values())==1 and sum(x=="valid" for x in states.values())==15
    states=c.verify(fragments+[fragments[2]],TOKEN,SOURCE,KEY);assert states[2]=="manipulated"
@pytest.mark.parametrize("tag_bits,minimum",[(64,8),(128,16)])
def test_distributed_global_reconstruction_threshold(tag_bits,minimum):
    c=DistributedGlobalMAC(tag_bits);fragments=c.issue(TOKEN,SOURCE,KEY);assert len(c.reconstruct_authenticator(fragments[:minimum]))==tag_bits//8
    with pytest.raises(ValueError):c.reconstruct_authenticator(fragments[:minimum-1])
def test_rs16_8_auth_shares_correct_errors_and_erasures():
    c=DistributedGlobalMAC(64);fragments=c.issue(TOKEN,SOURCE,KEY);expected=c.reconstruct_authenticator(fragments);assert c.reconstruct_authenticator(fragments[:8])==expected
    changed=fragments.copy()
    for i in range(4):f=changed[i];changed[i]=AuthFragment(f.index,f.symbol,f.share^1)
    assert c.reconstruct_authenticator(changed)==expected
def test_joint_code_recovers_unordered_with_four_missing():
    c=JointFragmentCode();fragments=c.issue(TOKEN,SOURCE,KEY);result=c.recover_and_verify(list(reversed(fragments[4:])),SOURCE,KEY);assert result["status"]=="valid" and result["token"]==TOKEN;assert sum(x=="missing" for x in result["states"].values())==4
def test_joint_code_mixed_valid_images_and_wrong_source_reject():
    c=JointFragmentCode();a=c.issue(TOKEN,SOURCE,KEY);other=ProvenanceToken(8,987654321,2);b=c.issue(other,b"source-B",KEY);mixed=a[:8]+b[8:];assert c.recover_and_verify(mixed,SOURCE,KEY)["status"]=="manipulated";assert c.recover_and_verify(a,b"wrong",KEY)["status"]=="manipulated"
def test_joint_code_insufficient_and_duplicate_explicit_states():
    c=JointFragmentCode();fragments=c.issue(TOKEN,SOURCE,KEY);result=c.recover_and_verify(fragments[:11],SOURCE,KEY);assert result["status"]=="insufficient" and sum(x=="missing" for x in result["states"].values())==5
    result=c.recover_and_verify(fragments+[fragments[0]],SOURCE,KEY);assert result["status"]=="manipulated" and result["states"][0]=="manipulated"
def test_replay_is_not_detected_without_external_freshness_registry():
    c=JointFragmentCode();fragments=c.issue(TOKEN,SOURCE,KEY);assert c.recover_and_verify(fragments,SOURCE,KEY)["status"]=="valid";assert c.recover_and_verify(fragments,SOURCE,KEY)["status"]=="valid"
def test_exhaustive_single_share_forgeries_match_exact_count():
    c=IndependentMAC(8);fragment=c.issue(TOKEN,SOURCE,KEY)[0];accepted=0
    for share in range(256):accepted+=c.verify([AuthFragment(0,fragment.symbol,share)],TOKEN,SOURCE,KEY)[0]=="valid"
    assert accepted==1
def test_exhaustive_small_case_erasures_permutations_and_mixes():
    c=IndependentMAC(8);fragments=c.issue(TOKEN,SOURCE,KEY)[:4]
    for mask in range(16):
        subset=[fragment for i,fragment in enumerate(fragments) if mask&(1<<i)];states=c.verify(subset,TOKEN,SOURCE,KEY);assert sum(x=="valid" for x in states.values())==len(subset)
    for permutation in itertools.permutations(fragments):assert sum(x=="valid" for x in c.verify(permutation,TOKEN,SOURCE,KEY).values())==4
    other=c.issue(ProvenanceToken(9,222,2),SOURCE,KEY)
    for split in range(5):assert sum(x=="manipulated" for x in c.verify(fragments[:split]+other[split:4],TOKEN,SOURCE,KEY).values())==4-split
def test_monte_carlo_is_deterministic_and_covers_six_controls():
    a=monte_carlo(JointFragmentCode(),120,5);b=monte_carlo(JointFragmentCode(),120,5);assert a==b;assert {int(key.split("_")[1]) for key in a}==set(range(6))
def test_comparison_exposes_128_bit_erasure_tradeoff_and_20_bit_recommendation_candidates():
    rows={x["construction"]:x for x in construction_comparison()};assert rows["global_hmac128_raw_split"]["minimum_auth_fragments"]==16;assert rows["joint_fragment_code64_rs16_8"]["bits_per_region"]==20 and rows["joint_fragment_code64_rs16_8"]["forged_identity_acceptance_at_12"]==2**-64
