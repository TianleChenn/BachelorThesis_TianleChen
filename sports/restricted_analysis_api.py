from __future__ import annotations

from itertools import combinations

import pandas as pd
from privacy.athlete_id import ATHLETE_ID_PATTERN

from .analysis import load_data, run_figure2, run_table1, run_table2
from .config import DISPLAY_NAMES, DOMAIN_ORDER, PREDICTORS
from .filters import apply_analysis_filters
from .analysis_noise_utility import evaluate_analysis_noise_utility

PUBLIC_DOMAIN_VARIABLES = set(PREDICTORS)
ALLOWED_METHODS = {
    "table1", "table2", "figure1", "figure2", "correlation",
    "variance_analysis", "individual_profile",
}
METHOD_ARGUMENT_SCHEMAS = {
    "individual_profile": {"required": {"subject_token", "variables"}, "optional": {"reference_group", "output_mode"}},
    "correlation": {"required": {"variables"}, "optional": {"filters", "method", "visualization"}},
    "figure1": {"required": {"variables"}, "optional": {"target", "group_field", "correlation_threshold", "variance_iterations", "filters"}},
    "figure2": {"required": {"variables"}, "optional": {"filters", "max_athletes", "reference_group"}},
    "variance_analysis": {"required": {"variables"}, "optional": {"group_field", "groups", "iterations", "filters", "visualization"}},
    "table1": {"required": set(), "optional": {"predictors", "variables", "target", "controls", "filters"}},
    "table2": {"required": set(), "optional": {"predictors", "variables", "filters", "group"}},
}
class RestrictedAnalysisAPI:
    def __init__(self, protected_dataframe=None, subject_reference=None):
        self.__df = protected_dataframe if protected_dataframe is not None else load_data()
        self.__subject_reference = subject_reference

    def __getattribute__(self, name):
        if name in {"df", "data", "dataset", "__df", "__dict__"}:
            raise AttributeError("Protected dataframe access is forbidden.")
        return object.__getattribute__(self, name)

    def _frame(self):
        return object.__getattribute__(self, "_RestrictedAnalysisAPI__df")

    def _variables(self, values, allowed=PUBLIC_DOMAIN_VARIABLES):
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            raise ValueError("Variables must be a non-empty list of public names.")
        if any(v not in allowed for v in values): raise ValueError("Raw or unsupported variable requested.")
        return values

    def _resolve_predictors(self, *, predictors=None, variables=None):
        selected = predictors if predictors is not None else variables
        return self._variables(selected)

    def _filtered(self, filters):
        return apply_analysis_filters(self._frame(), filters)

    @staticmethod
    def _safe_float(value):
        return None if pd.isna(value) else float(value)

    def individual_profile(self, *, subject_token: str, variables: list[str], reference_group: str="all", output_mode: str="standardized_profile") -> dict:
        if subject_token!="CURRENT_SUBJECT":raise ValueError("Only CURRENT_SUBJECT is accepted.")
        athlete_reference=object.__getattribute__(self,"_RestrictedAnalysisAPI__subject_reference")
        if not isinstance(athlete_reference,str) or not ATHLETE_ID_PATTERN.fullmatch(athlete_reference):raise ValueError("Trusted local subject context is unavailable.")
        if reference_group!="all" or output_mode!="standardized_profile":raise ValueError("Invalid individual profile mode.")
        variables=self._variables(variables)
        if variables!=DOMAIN_ORDER:raise ValueError("Individual profile must use the eight domains in canonical order.")
        dataframe=self._frame()
        identifiers=dataframe["athlete_id"].astype(str)
        matches=dataframe[identifiers.str.casefold()==athlete_reference.casefold()]
        if matches.empty:raise ValueError("Trusted local subject was not found.")
        numeric=dataframe[DOMAIN_ORDER].apply(pd.to_numeric,errors="coerce")
        means=numeric.mean(axis=0);standard_deviations=numeric.std(axis=0,ddof=0).replace(0,1)
        values=pd.to_numeric(matches.iloc[0][DOMAIN_ORDER],errors="coerce")
        z_scores=(values-means)/standard_deviations
        elite_mask=pd.to_numeric(dataframe["expertise_value"],errors="coerce").ge(13)
        elite_z_scores=(numeric.loc[elite_mask]-means)/standard_deviations
        elite_mean=elite_z_scores.mean(axis=0)
        elite_mean_profile=[None if pd.isna(elite_mean.get(domain)) else float(elite_mean[domain])
            for domain in DOMAIN_ORDER]
        elite_count=int(elite_mask.sum())
        profile=[];missing=[]
        for domain in DOMAIN_ORDER:
            value=z_scores.get(domain)
            value=None if pd.isna(value) else max(-3.0,min(3.0,float(value)))
            if value is None:missing.append(domain)
            interpretation=("Unavailable" if value is None else "Well above average" if value>=1.5
                else "Above average" if value>=.5 else "Near average" if value>-.5
                else "Below average" if value>-1.5 else "Well below average")
            profile.append({"variable":DISPLAY_NAMES.get(domain,domain.replace("_"," ").title()),
                "domain_key":domain,"z_score":value,"interpretation":interpretation})
        if len(profile)!=8:raise RuntimeError("The individual profile must contain exactly eight domains.")
        available=[row for row in profile if row["z_score"] is not None]
        strongest=sorted(available,key=lambda row:row["z_score"],reverse=True)[:3]
        weakest=sorted(available,key=lambda row:row["z_score"])[:3]
        paper_pattern=all(z_scores.get(domain)>0 for domain in ("basic_cognitive_function","lower_body_dynamics","blood_micronutrients"))
        profile_label="Anonymous Profile"
        from .figures import generate_individual_profile_line_figure
        result={"analysis":"Anonymous Athlete Profile","profile_label":profile_label,
            "profile":profile,"table":profile,
            "figure":generate_individual_profile_line_figure(profile,profile_label=profile_label,
                elite_mean_profile=elite_mean_profile),
            "reference_type":"elite_mean","reference_label":"Elite Mean Profile",
            "elite_reference_count":elite_count,"elite_mean_profile":elite_mean_profile,
            "strongest_domains":strongest,"weakest_domains":weakest,
            "paper_like_three_domain_pattern":bool(paper_pattern),
            "domain_count":8,"domain_keys":list(DOMAIN_ORDER),"raw_values_included":False,"identifier_exposed":False,
            "output_type":"individual_standardized_profile","missing_domains":missing,
            "safe_warning":"Unavailable domains are retained and shown without fabricated values." if missing else None,
            "cohort_size":len(self._frame())}
        result["noise_utility"]=evaluate_analysis_noise_utility(
            "individual_profile",
            dataframe,
            variables=list(DOMAIN_ORDER),
            subject_index=matches.index[0],
        )
        return result

    def table1(self, *, predictors=None, variables=None, target=None, controls=None, filters=None):
        self._resolve_predictors(predictors=predictors, variables=variables)
        if controls is not None and not isinstance(controls, list):
            raise ValueError("controls must be a list when provided.")
        dataframe=self._filtered(filters)
        outcome=pd.to_numeric(dataframe["elite_status"],errors="coerce").dropna()
        if outcome.nunique()<2:
            raise ValueError(
                "Table 1 logistic regression is not statistically applicable to the "
                "selected cohort because both elite and semi-elite outcome classes are required."
            )
        result=run_table1(dataframe=dataframe)
        if not any(
            isinstance(model,dict) and model.get("converged") is True
            for model in result.get("model_stats",[])
        ):
            raise ValueError(
                "Table 1 logistic regression is not statistically applicable to the "
                "selected cohort because none of the four model specifications converged."
            )
        result["noise_utility"]=evaluate_analysis_noise_utility("table1",dataframe,variables=list(PREDICTORS))
        result["filters"]=dict(filters or {})
        result["cohort_size"]=len(dataframe)
        return result

    def table2(self, *, predictors=None, variables=None, filters=None, group=None):
        self._resolve_predictors(predictors=predictors, variables=variables)
        dataframe=self._filtered(filters)
        result=run_table2(group="all",dataframe=dataframe)
        result["noise_utility"]=evaluate_analysis_noise_utility("table2",dataframe,variables=list(PREDICTORS),group="all")
        result["filters"]=dict(filters or {})
        return result
    def figure1(self, *, variables, target="expertise_value", group_field="elite_status", correlation_threshold=.15, variance_iterations=1000, filters=None):
        self._variables(variables)
        if target!="expertise_value" or group_field!="elite_status":raise ValueError("Figure 1 target/group field is not allowed.")
        if not isinstance(correlation_threshold,(int,float)) or not 0<=correlation_threshold<=1:raise ValueError("correlation_threshold must be between 0 and 1.")
        if not isinstance(variance_iterations,int) or not 100<=variance_iterations<=2000:raise ValueError("variance_iterations must be 100-2000.")
        from .figures import create_figure1,figure1_summary_table
        dataframe=self._filtered(filters)
        rows=figure1_summary_table(dataframe=dataframe,variables=variables,variance_iterations=variance_iterations)
        result={"analysis":"Figure 1-style group statistics","filters":dict(filters or {}),
            "figure":create_figure1(dataframe=dataframe,variables=variables,
                correlation_threshold=correlation_threshold,variance_iterations=variance_iterations),
            "table":rows,"rows":rows,"cohort_size":len(dataframe)}
        result["noise_utility"]=evaluate_analysis_noise_utility("figure1",dataframe,variables=variables,correlation_threshold=correlation_threshold,variance_iterations=variance_iterations)
        return result
    def figure2(self, *, variables, filters=None, max_athletes=50, reference_group="selected_cohort", group=None):
        self._variables(variables)
        if reference_group not in {"selected_cohort","selected","all"}:raise ValueError("Invalid Figure 2 reference group.")
        if max_athletes is not None and max_athletes not in {10,20,50,80}:raise ValueError("max_athletes must be 10, 20, 50, 80, or None.")
        result=run_figure2(variables=variables,filters=filters,max_athletes=max_athletes,
            reference_group=reference_group,dataframe=self._frame(),group=group)
        result["noise_utility"]=evaluate_analysis_noise_utility("figure2",self._frame(),variables=variables,filters=filters,max_athletes=max_athletes,reference_group=reference_group)
        return result
    def correlation(self, *, variables, filters=None, method="pearson", visualization=True):
        variables=self._variables(variables)
        if method not in {"pearson","spearman"}:raise ValueError("method must be 'pearson' or 'spearman'.")
        dataframe=self._filtered(filters)
        numeric=dataframe[variables].apply(pd.to_numeric,errors="coerce")
        rows=[]
        for left,right in combinations(variables,2):
            pair=numeric[[left,right]].dropna()
            coefficient=self._safe_float(pair[left].corr(pair[right],method=method)) if len(pair)>=3 else None
            rows.append({"variable_1":left,"variable_2":right,"correlation":coefficient,
                "r":coefficient,"method":method,"n":len(pair)})
        result={"analysis":"Correlation analysis","method":method,"table":rows,"rows":rows,
            "correlation_matrix":numeric.corr(method=method),"filters":dict(filters or {}),
            "cohort_size":len(dataframe)}
        # Keep the visualization argument for the restricted-call schema, but return
        # correlation results as a table only. The frontend must not render a heatmap
        # for the original Correlation result.
        result["noise_utility"]=evaluate_analysis_noise_utility(
            "correlation",dataframe,variables=variables,method=method)
        return result
    def variance_analysis(self, *, variables, group_field="elite_status", groups=None, iterations=1000, filters=None, visualization=True):
        self._variables(variables);groups=groups or ["elite","semi_elite"]
        if group_field!="elite_status" or groups!=["elite","semi_elite"]:raise ValueError("Invalid variance groups.")
        if not isinstance(iterations,int) or not 100<=iterations<=2000:raise ValueError("iterations must be 100-2000.")
        dataframe=self._filtered(filters)
        expertise=pd.to_numeric(dataframe["expertise_value"],errors="coerce")
        elite_size=int(expertise.ge(13).sum());semi_size=int(expertise.lt(13).sum())
        if elite_size<2 or semi_size<2:
            raise ValueError(
                "Variance analysis requires at least two elite and two semi-elite athletes "
                "in the selected cohort."
            )
        result={"analysis":"Dynamic variance analysis","table":[],"filters":dict(filters or {}),
            "cohort_size":len(dataframe),"elite_group_size":elite_size,"semi_elite_group_size":semi_size}
        for variable in variables:
            elite=dataframe[dataframe[group_field].astype(int)==1][variable]
            semi=dataframe[dataframe[group_field].astype(int)==0][variable]
            result["table"].append({"variable":variable,"elite_variance":self._safe_float(elite.var()),"semi_elite_variance":self._safe_float(semi.var()),"iterations":iterations})
        if visualization:
            from .figures import create_variance_plot
            result["figure"]=create_variance_plot(iterations=iterations,dataframe=dataframe)
        result["noise_utility"]=evaluate_analysis_noise_utility("variance_analysis",dataframe,variables=variables,iterations=iterations)
        return result
