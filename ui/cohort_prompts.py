"""Safe, human-readable dashboard prompts shared by UI and diagnostics."""
from __future__ import annotations


def build_dashboard_prompt(analysis: str, cohort_prompt_text: str, figure2_size_option: str = "50") -> str:
    if analysis == "table1":
        return ("Generate the four logistic regression models corresponding to "
            f"Table 1 for {cohort_prompt_text} using all eight public domains, "
            "with elite_status as the binary target. Use four model specifications: "
            "no controls, sex, age, and both sex and age.")
    if analysis == "table2":
        return ("Generate the four multiple linear regression models corresponding "
            f"to Table 2 for {cohort_prompt_text} using all eight public domains, "
            "with expertise_value as the continuous target.")
    if analysis == "figure1":
        return (f"Generate Figure 1 for {cohort_prompt_text} using all eight public domains, "
            "expertise_value as target, elite_status as the analysis grouping field, "
            "correlation threshold 0.15, and 1000 comparison-group variance samples.")
    if analysis == "figure2":
        if figure2_size_option == "All":
            return ("Generate Figure 2-style standardized z-score profiles for "
                f"{cohort_prompt_text}, showing all available anonymous athletes and using "
                "the selected cohort as the comparison reference.")
        return ("Generate Figure 2-style standardized z-score profiles for "
            f"{cohort_prompt_text}, showing at most {figure2_size_option} anonymous athletes "
            "and using the selected cohort as the comparison reference.")
    if analysis == "correlation":
        return ("Calculate pairwise Pearson correlations among all eight public "
            f"domains for {cohort_prompt_text}.")
    if analysis == "variance_analysis":
        return (f"Run variance analysis for {cohort_prompt_text} across all eight public domains, "
            "using elite_status as the grouping field. Compare the elite and semi-elite "
            "groups using 1000 repeated comparison samples.")
    raise ValueError("Unsupported dashboard analysis.")
