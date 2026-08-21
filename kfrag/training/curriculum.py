PHASES=("natural_image_communication","full_packets","benign_robustness","partial_evidence","geometric_synchronization","manipulation_splicing")
def may_progress(phase,gate_results,manual_override=False):
    if phase not in PHASES: raise ValueError("unknown curriculum phase")
    return bool(manual_override or gate_results.get("passed",False))
def next_phase(phase,gate_results,manual_override=False):
    if not may_progress(phase,gate_results,manual_override): raise RuntimeError("phase gate failed; scientific progression blocked")
    i=PHASES.index(phase); return None if i+1==len(PHASES) else PHASES[i+1]
