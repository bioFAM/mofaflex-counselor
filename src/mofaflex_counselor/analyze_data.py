def ____analyze_data(
    data_var_name: str, data_type: str, path: str | None, featuresets_var_name: str | None, return_json: bool = True
):
    import json
    import operator
    from contextlib import suppress
    from functools import reduce
    from pathlib import Path

    import anndata as ad
    import mudata as md
    import pandas as pd

    ret = {"data_var_name": data_var_name, "featuresets_var_name": featuresets_var_name}
    data = None
    try:
        data = globals()[data_var_name]
    except KeyError:
        if path is not None:  # TODO: use anndata.experimental.read_lazy for sufficiently recent anndata versions
            path = Path(path)
            if path.suffix in (".h5ad", ".h5mu"):
                data = md.read(path, backed=True)
            else:
                data = md.read_zarr(path)

    if not isinstance(data, ad.AnnData | md.MuData):
        for var_name, data in reversed(globals().items()):
            if isinstance(data, ad.AnnData | md.MuData):
                ret["data_var_name"] = var_name
                break
        else:
            raise RuntimeError("No AnnData or MuData object found.")

    def get_floating_cols(df):
        return df.select_dtypes("float").columns.to_list()

    covariates_obs_cols = get_floating_cols(data.obs)

    def get_floating_entries(kvstore, exclude=()):
        return [
            k
            for k, v in kvstore.items()
            if k not in exclude
            and (isinstance(v, pd.DataFrame) and v.select_dtypes("float").shape[1] == v.shape[1] or v.dtype.kind == "f")
        ]

    exclude_cols = ()
    if isinstance(data, md.MuData):
        exclude_cols = data.mod.keys()

    covariates_obsm_keys = get_floating_entries(data.obsm, exclude=exclude_cols)

    def get_bool_entries(kvstore, exclude=()):
        return [
            k
            for k, v in kvstore.items()
            if k not in exclude
            and (isinstance(v, pd.DataFrame) and v.select_dtypes("bool").shape[1] == v.shape[1] or v.dtype.kind == "b")
        ]

    annotations_varm_keys = get_bool_entries(data.varm, exclude=exclude_cols)

    ret["n_obs"] = data.n_obs
    if isinstance(data, md.MuData):
        ret["type"] = "MuData"
        ret["n_views"] = len(data.mod)
        ret["n_vars"] = {modname: mod.n_vars for modname, mod in data.mod.items()}
        ret["X_nonnegative"] = all((mod.X >= 0).all() for mod in data.mod.values())
        ret["layers"] = [
            {"name": layer, "nonnegative": bool(all((mod.layers[layer] >= 0).all()) for mod in data.mod.values())}
            for layer in reduce(operator.and_, (mod.layers.keys() for mod in data.mod.values()))
        ]
        ret["grouping_cols"] = data.obs.select_dtypes(exclude="float").columns.to_list()

        for mod in data.mod.values():
            covariates_obs_cols.extend(get_floating_cols(mod.obs))
            covariates_obsm_keys.extend(get_floating_entries(mod.obsm))
            annotations_varm_keys.extend(k for k in get_bool_entries(mod.varm))
    else:
        ret["type"] = "AnnData"
        ret["n_views"] = 1
        ret["n_vars"] = data.n_vars
        ret["X_nonnegative"] = bool((data.X >= 0).all())
        ret["layers"] = [
            {"name": lname, "nonnegative": bool((layer >= 0).all())} for lname, layer in data.layers.items()
        ]
        ret["grouping_cols"] = []

    ret["covariates_obs_cols"] = covariates_obs_cols
    ret["covariates_obsm_keys"] = covariates_obsm_keys
    ret["annotations_varm_keys"] = annotations_varm_keys

    featuresets = None
    with suppress(KeyError):
        featuresets = globals()[featuresets_var_name]
    if featuresets.__class__.__name__ != "FeatureSets" or not featuresets.__class__.__module__.startswith("mofaflex."):
        for var_name, featuresets in reversed(globals().items()):
            if featuresets.__class__.__name__ == "FeatureSets" and featuresets.__class__.__module__.startswith(
                "mofaflex."
            ):
                ret["featuresets_var_name"] = var_name
                break

    if featuresets is None:
        ret["featuresets_var_name"] = None

    return ret if not return_json else json.dumps(ret)
