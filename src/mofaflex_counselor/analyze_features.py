def ____optimize_features_filtering(data_var_name: str, featuresets_var_name: str, return_json: bool = True):
    import json

    orig_features = features = globals()[featuresets_var_name]
    names = globals()[data_var_name].var_names

    if len(features) < 200:
        features = None
    else:
        min_fraction, min_fraction_diff = 0, 0.2
        min_count, min_count_diff = 5, 10
        max_count, max_count_diff = 300, 50

        last_sign = 1
        i = 0
        while not 50 <= len(features) <= 200 and i < 10:
            features = orig_features.filter(names, min_fraction=min_fraction, min_count=min_count, max_count=max_count)
            if len(features) < 50:
                min_fraction = max(0, min_fraction - min_fraction_diff)
                min_count = max(0, min_count - min_count_diff)
                max_count += max_count_diff
                sign = -1
            elif len(features) > 200:
                min_fraction += min_fraction_diff
                min_count += max_count_diff
                max_count = max(0, max_count - max_count_diff)
                sign = 1

            if last_sign != sign:
                min_fraction_diff *= 0.5
                min_count_diff //= 2
                max_count_diff //= 2
            last_sign = sign
            i += 1

        if not 50 <= len(features) <= 200:
            features = None

    if features is None:
        return None if not return_json else json.dumps(None)
    else:
        ret = {"min_fraction": min_fraction, "min_count": min_count, "max_count": max_count}
        return ret if not return_json else json.dumps(ret)
