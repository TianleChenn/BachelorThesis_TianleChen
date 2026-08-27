from __future__ import annotations
import inspect
import logging
from typing import Any
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from sports.config import DOMAIN_ORDER

logger=logging.getLogger(__name__)
SAFE_TABLE_KEYS=("table","rows","profile","summary_table")
NOISE_DASHBOARD_VERSION="noise-dashboard-all-analyses-v2"

NOISE_ANALYSIS_PRESENTATION={
    "table1":{
        "original_title":"Original Table 1-style Logistic Regression",
        "mean_title":"Mean Perturbed Table 1-style Logistic Regression",
        "evaluation_title":"Logistic Regression Noise Stability Evaluation",
        "metric_label":"Average Logistic Coefficient Difference",
        "stability_title":"Logistic Coefficient Stability After Numerical Perturbation",
    },
    "table2":{
        "original_title":"Original Table 2-style Linear Regression",
        "mean_title":"Mean Perturbed Table 2-style Linear Regression",
        "evaluation_title":"Linear Regression Noise Stability Evaluation",
        "metric_label":"Average Standardized Coefficient Difference",
        "stability_title":"Standardized Coefficient Stability After Numerical Perturbation",
    },
    "figure1":{
        "original_title":"Original Figure 1",
        "mean_title":"Mean Perturbed Figure 1",
        "evaluation_title":"Figure 1 Noise Stability Evaluation",
        "metric_label":"Average Figure 1 Coefficient Difference",
        "stability_title":"Figure 1 Coefficient Stability After Numerical Perturbation",
    },
    "figure2":{
        "original_title":"Original Figure 2",
        "mean_title":"Mean Perturbed Figure 2",
        "evaluation_title":"Figure 2 Noise Stability Evaluation",
        "metric_label":"Average Profile Difference",
        "stability_title":"Figure 2 Profile Stability After Numerical Perturbation",
    },
    "correlation":{
        "original_title":"Original Correlation Result",
        "mean_title":"Mean Perturbed Correlation Result",
        "evaluation_title":"Correlation Noise Stability Evaluation",
        "metric_label":"Average Correlation Difference",
        "stability_title":"Correlation Stability After Numerical Perturbation",
    },
    "variance_analysis":{
        "original_title":"Original Variance Analysis",
        "mean_title":"Mean Perturbed Variance Analysis",
        "evaluation_title":"Variance Noise Stability Evaluation",
        "metric_label":"Average Variance Difference",
        "stability_title":"Variance Stability After Numerical Perturbation",
    },
    "individual_profile":{
        "original_title":"Original Anonymous Athlete Profile",
        "mean_title":"Mean Perturbed Anonymous Athlete Profile",
        "evaluation_title":"Anonymous Profile Noise Stability Evaluation",
        "metric_label":"Average Anonymous Profile Difference",
        "stability_title":"Anonymous Profile Stability After Numerical Perturbation",
    },
}

def get_noise_analysis_presentation(analysis_key:str)->dict[str,str]:
    """Return fixed labels for a supported analysis noise experiment."""
    return NOISE_ANALYSIS_PRESENTATION.get(
        str(analysis_key or ""),
        {
            "original_title":"Original Analysis Result",
            "mean_title":"Mean Perturbed Analysis Result",
            "evaluation_title":"Noise Stability Evaluation",
            "metric_label":"Average Difference",
            "stability_title":"Original vs Mean Perturbed Stability",
        },
    )


def _render_mean_perturbed_result(mean_result:dict)->bool:
    """Render the mean figure before its supporting table when both are present."""
    if not isinstance(mean_result,dict):
        return False
    table=mean_result.get("table")
    figure=mean_result.get("figure")
    rendered=False
    if hasattr(figure,"savefig"):
        st.pyplot(figure,use_container_width=True,clear_figure=False)
        rendered=True
    if isinstance(table,pd.DataFrame):
        st.dataframe(table,use_container_width=True,hide_index=True)
        rendered=True
    elif isinstance(table,(list,tuple)):
        st.dataframe(pd.DataFrame(table),use_container_width=True,hide_index=True)
        rendered=True
    return rendered


def render_noise_analysis_section(
    utility:dict,
    *,
    analysis_key:str,
) -> None:
    """Render only Parts 2 and 3 of the unified dashboard noise result.

    The caller renders the original result first. Keeping this function focused
    on the two noise-derived parts prevents the result from being rendered twice.
    """
    presentation=get_noise_analysis_presentation(analysis_key)

    mean_result=utility.get("mean_perturbed_result")
    if not isinstance(mean_result,dict):
        # Compatibility with in-memory results created before the unified schema.
        mean_result={
            "title":utility.get("noisy_result_title"),
            "figure":utility.get("noisy_result_figure"),
        }
    st.caption("PART 2")
    st.markdown(f"### {presentation['mean_title']}")
    if not _render_mean_perturbed_result(mean_result):
        st.warning("Mean perturbed analysis result is unavailable. Run the analysis again to regenerate the noise experiment.")

    metric=utility.get("average_difference")
    if not isinstance(metric,dict):
        metric=utility.get("primary_metric") or {}
    value=metric.get("value")
    unit=str(metric.get("display_unit") or "").strip()
    display="N/A" if value is None else f"{float(value):.4f}"
    if display!="N/A" and unit:
        display=f"{display} {unit}"

    st.caption("PART 3")
    st.markdown(f"### {presentation['evaluation_title']}")
    # Public labels are fixed per analysis so a variance label can never leak
    # into the other five dashboards through stale session data.
    st.metric(presentation["metric_label"],display)
    if metric.get("explanation"):
        st.write(metric["explanation"])

    stability=utility.get("stability_figure")
    st.markdown(f"#### {presentation['stability_title']}")
    if hasattr(stability,"savefig"):
        st.pyplot(stability,use_container_width=True,clear_figure=False)
    else:
        st.warning("Stability comparison data are unavailable. Run the analysis again.")


def _render_noise_utility(utility:dict,baseline_result=None)->None:
    """Backward-compatible wrapper around the unified noise renderer."""
    render_noise_analysis_section(
        utility,
        analysis_key=str(utility.get("analysis_key") or ""),
    )


def render_noise_utility(utility:dict,baseline_result=None)->None:
    if isinstance(utility,dict):
        _render_noise_utility(utility,baseline_result)
def _unsupported(value,source):
    logger.warning("Unsupported frontend object suppressed: type=%s module=%s renderer=%s",type(value).__name__,getattr(type(value),"__module__","unknown"),source)
def render_analysis_result(value:Any)->None:
    if value is None:return None
    if isinstance(value,DeltaGenerator) or inspect.ismodule(value) or inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value) or callable(value):
        _unsupported(value,"render_analysis_result");return None
    if hasattr(value,"savefig"):
        st.pyplot(value,use_container_width=True);return None
    if isinstance(value,str):st.markdown(value);return None
    if isinstance(value,pd.DataFrame):
        st.dataframe(value,use_container_width=True,hide_index=True);return None
    if isinstance(value,dict):
        figure=value.get("figure")
        if isinstance(figure,DeltaGenerator) or inspect.ismodule(figure) or inspect.isclass(figure) or callable(figure):_unsupported(figure,"render_analysis_result.figure")
        elif hasattr(figure,"savefig"):st.pyplot(figure,use_container_width=True,clear_figure=False)
        rendered=False
        for key in SAFE_TABLE_KEYS:
            rows=value.get(key)
            if isinstance(rows,(list,tuple)):
                frame=pd.DataFrame(rows)
                if value.get("output_type")=="individual_standardized_profile":
                    if "domain_key" not in frame.columns:
                        st.error("The individual profile response does not contain the required domain keys.");return None
                    order={key:index for index,key in enumerate(DOMAIN_ORDER)}
                    frame["_order"]=frame["domain_key"].map(order);frame=frame.sort_values("_order").drop(columns=["_order"])
                    returned=frame["domain_key"].tolist()
                    if returned!=DOMAIN_ORDER:
                        missing=[key for key in DOMAIN_ORDER if key not in returned]
                        st.error("The individual profile is incomplete. Missing: "+", ".join(missing));return None
                if {"variable","z_score","interpretation"}.issubset(frame.columns):
                    frame=frame[["variable","z_score","interpretation"]].rename(columns={"variable":"Variable","z_score":"z-score","interpretation":"Interpretation"})
                if not frame.empty:st.dataframe(frame,use_container_width=True,hide_index=True);rendered=True
                break
            if isinstance(rows,pd.DataFrame):st.dataframe(rows,use_container_width=True,hide_index=True);rendered=True;break
        if figure is None and not rendered:st.warning("No table output available.")
        if value.get("output_type")=="individual_standardized_profile":
            rows=value.get("table") or []
            with st.expander("Profile rendering diagnostics",expanded=False):
                st.json({"expected_domain_count":8,"domain_count":value.get("domain_count"),
                    "domain_keys_returned":[row.get("domain_key") for row in rows if isinstance(row,dict)],
                    "figure_available":figure is not None,
                    "figure_type":type(figure).__name__ if figure is not None else None,
                    "table_row_count":len(rows),"output_type":value.get("output_type")})
        return None
    _unsupported(value,"render_analysis_result");st.warning("The analysis result could not be displayed.");return None
