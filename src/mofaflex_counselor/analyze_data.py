def ____analyze_data(var_name: str, data_type: str, path: str | None, return_json: bool = True):
    import json
    import operator
    from functools import reduce
    from pathlib import Path

    import anndata as ad
    import mudata as md
    import pandas as pd

    try:
        data = globals()[var_name]
    except KeyError:
        if path is not None:  # TODO: use anndata.experimental.read_lazy for sufficiently recent anndata versions
            path = Path(path)
            if path.suffix in (".h5ad", ".h5mu"):
                data = md.read(path, backed=True)
            else:
                data = md.read_zarr(path)
        else:
            raise
    if not isinstance(data, ad.AnnData | md.MuData):
        raise TypeError("Need AnnData or MuData.")

    ret = {}

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

    covariates_obsm_keys = get_floating_entries(data.obsm, exclude=data.mod.keys())

    def get_bool_entries(kvstore, exclude=()):
        return [
            k
            for k, v in kvstore.items()
            if k not in exclude
            and (isinstance(v, pd.DataFrame) and v.select_dtypes("bool").shape[1] == v.shape[1] or v.dtype.kind == "b")
        ]

    annotations_varm_keys = get_bool_entries(data.varm, exclude=data.mod.keys())

    ret["n_obs"] = data.n_obs
    if isinstance(data, md.MuData):
        ret["type"] = "MuData"
        ret["n_views"] = len(data.mod)
        ret["n_vars"] = {modname: mod.n_vars for modname, mod in data.mod.items()}
        ret["layers"] = list(reduce(operator.and_, (mod.layers.keys() for mod in data.mod.values())))
        ret["grouping_cols"] = data.obs.select_dtypes(exclude="float").columns.to_list()

        for mod in data.mod.values():
            covariates_obs_cols.extend(get_floating_cols(mod.obs))
            covariates_obsm_keys.extend(get_floating_entries(mod.obsm))
            annotations_varm_keys.extend(k for k in get_bool_entries(mod.varm))
    else:
        ret["type"] = "AnnData"
        ret["n_views"] = 1
        ret["n_vars"] = data.n_vars
        ret["layers"] = list(data.layers.keys())
        ret["grouping_cols"] = []

    ret["covariates_obs_cols"] = covariates_obs_cols
    ret["covariates_obsm_keys"] = covariates_obsm_keys
    ret["annotations_varm_keys"] = annotations_varm_keys

    return ret if not return_json else json.dumps(ret)
