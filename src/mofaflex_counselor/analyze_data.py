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

    if isinstance(data, md.MuData):
        grouping_cols = data.obs.select_dtypes(exclude="float").columns.to_list()
    else:
        grouping_cols = []

    def get_floating_cols(df):
        return df.select_dtypes("float").columns.to_list()

    covariates_obs_cols = get_floating_cols(data.obs)
    if isinstance(data, md.MuData):
        for mod in data.mod.values():
            covariates_obs_cols.extend(get_floating_cols(mod.obs))

    def get_floating_entries(kvstore):
        return [
            k
            for k, v in kvstore.items()
            if isinstance(v, pd.DataFrame) and v.select_dtypes("float").shape[1] == v.shape[1] or v.dtype.kind == "f"
        ]

    covariates_obsm_keys = get_floating_entries(data.obsm)
    if isinstance(data, md.MuData):
        for mod in data.mod.values():
            covariates_obsm_keys.extend(get_floating_entries(mod.obsm))

    def get_bool_entries(kvstore):
        return [
            k
            for k, v in kvstore.items()
            if isinstance(v, pd.DataFrame) and v.select_dtypes("bool").shape[1] == v.shape[1] or v.dtype.kind == "b"
        ]

    annotation_varm_keys = get_bool_entries(data.varm)
    if isinstance(data, md.MuData):
        for mod in data.mod.values():
            annotation_varm_keys.extend(get_bool_entries(mod.varm))

    if isinstance(data, ad.AnnData):
        layers = list(data.layers.keys())
    else:
        layers = list(reduce(operator.and_, (mod.layers.keys() for mod in data.mod.values())))

    ret = {
        "grouping_cols": grouping_cols,
        "covariates_obs_cols": covariates_obs_cols,
        "covariates_obsm_keys": covariates_obsm_keys,
        "annotations_varm_keys": annotation_varm_keys,
        "layers": layers,
    }
    return ret if not return_json else json.dumps(ret)
