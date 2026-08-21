STATES=("valid_authenticated","invalid_authentication","missing_or_unobserved","duplicate_or_conflicting","undecodable")
def protocol_evidence_map(candidates,accepted,rejected,conflicts):
    result=["missing_or_unobserved"]*16
    for i in accepted: result[i]="valid_authenticated"
    for i in conflicts: result[i]="duplicate_or_conflicting"
    for candidate,reason in rejected:
        if candidate.packet is not None and result[candidate.packet.index]=="missing_or_unobserved": result[candidate.packet.index]=reason
        elif candidate.observed:
            y=min(3,max(0,int(candidate.location[1]*4))); x=min(3,max(0,int(candidate.location[0]*4)))
            result[y*4+x]="undecodable"
    return [result[i:i+4] for i in range(0,16,4)]

def experimental_map(protocol_map,ground_truth=None,manipulation_probabilities=None):
    flat=sum(protocol_map,[]); output=[]
    for i,state in enumerate(flat):
        if state=="valid_authenticated": label="valid"
        elif ground_truth is not None and ground_truth[i] and state=="invalid_authentication": label="manipulated"
        elif state=="missing_or_unobserved": label="missing"
        else: label="uncertain"
        output.append(label)
    return [output[i:i+4] for i in range(0,16,4)]
