from collections import namedtuple
from collections.abc import Sequence
from copy import copy
from typing import Annotated, Literal

from pydantic import Field, PositiveInt, create_model

from .notebook import DataAnalysisResult

_ConditionalParameterAnnotation = namedtuple("ConditionalParameterAnnotation", ["default_typehint", "field"])

_parameters_always = {
    "n_factors": Annotated[
        PositiveInt,
        Field(
            default=10,
            description="Number of latent factors. More factors increase training time and memory requirements. Downstream analysis typically focuses on the 5-10 most important factors. Highly complex datasets may require a large number of factors.",
        ),
    ],
    "factor_prior": Annotated[
        Literal["Normal", "Laplace", "Horseshoe", "SnS"],
        Field(
            default="Normal",
            description="Prior distribution to use for the factors. Laplace, Horseshoe and SnS are sparsity-inducing distributions that can yield better interpretability of the results. SnS typically yields the most sparse results. The Horseshoe does not achieve exact zeros, but it also does not shrink values that are far from zero. The Laplace shrinks all values.",
        ),
    ],
    "weight_prior": Annotated[
        Literal["Normal", "Laplace", "Horseshoe", "SnS"],
        Field(
            default="Normal",
            description="Prior distribution to use for the weights. Laplace, Horseshoe and SnS are sparsity-inducing distributions that can yield better interpretability of the results. SnS typically yields the most sparse results. The Horseshoe does not achieve exact zeros, but it also does not shrink values that are far from zero. The Laplace shrinks all values.",
        ),
    ],
    "nonnegative_factors": Annotated[
        bool,
        Field(
            default=False,
            description="Constrain the factors to be nonnegative. This may improve interpretability in some cases.",
        ),
    ],
    "nonnegative_weights": Annotated[
        bool,
        Field(
            default=False,
            description="Constrain the weights to be nonnegative. This may improve interpretability in some cases.",
        ),
    ],
}

_parameters_conditional = {
    "data_type": _ConditionalParameterAnnotation(
        Literal["AnnData", "MuData"], Field(description="Type of the input data.")
    ),
    "layer": _ConditionalParameterAnnotation(
        str | None,
        Field(
            default=None,
            description="Name of the layer in the input file to use. If omitted the X matrix will be used.",
        ),
    ),
    "group_by": _ConditionalParameterAnnotation(
        str | Sequence[str] | None,
        Field(default=None, description="Columns of .obs in the MuData to group the data by."),
    ),
    "annotations_varm_key": _ConditionalParameterAnnotation(
        str | None,
        Field(
            default=None,
            description="Key of the .varm attribute of the data object that contains gene set annotations for the Horseshoe prior. Will be ignored if the weight prior is not Horseshoe. If the weight prior is Horseshoe and this parameter is set, the annotated gene sets will be used as prior information in the Horseshoe prior.",
        ),
    ),
}


def make_mofaflex_parameters_model(analysis_result: DataAnalysisResult | None):
    parameters_always = _parameters_always
    if analysis_result is None:
        parameters_conditional = _parameters_conditional.copy()
        groupby = _parameters_conditional["group_by"]
        field = copy(groupby.field)
        field.description += " Only useful if the input is a MuData file. Ignored for AnnData files."
        parameters_conditional["group_by"] = _ConditionalParameterAnnotation(groupby.default_typehint, field)
        conditional_params = {
            param_name: Annotated[param.default_typehint, param.field]
            for param_name, param in parameters_conditional.items()
        }
    else:
        conditional_params = {}
        if analysis_result.layers:
            conditional_params["layer"] = Annotated[
                Literal[*analysis_result.layers] | None, _parameters_conditional["layer"].field
            ]
        if analysis_result.grouping_cols:
            conditional_params["group_by"] = Annotated[
                Literal[*analysis_result.grouping_cols] | Sequence[Literal[*analysis_result.grouping_cols]] | None,
                _parameters_conditional["group_by"].field,
            ]
        if analysis_result.annotations_varm_keys:
            conditional_params["annotations_varm_key"] = Annotated[
                Literal[*analysis_result.annotations_varm_keys] | None,
                _parameters_conditional["annotations_varm_key"].field,
            ]
            parameters_always = parameters_always.copy()
            parameters_always["weight_prior"] = copy(parameters_always["weight_prior"])
            parameters_always["weight_prior"].__metadata__[
                0
            ].description += " The Horseshoe uses the gene sets defined by annotations_varm_key as prior information."
    return create_model("MofaFlexParameters", **_parameters_always, **conditional_params)
