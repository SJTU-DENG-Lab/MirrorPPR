def QwenImageMAEStateDictConverter(state_dict):
    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        if k.startswith("vit."):
            k = k.replace("vit.", "model.")
            state_dict_[k] = v
    return state_dict_
