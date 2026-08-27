"""Generate the locked 100-request v2 preference-training candidate set."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.athlete_router_evaluation_common import load_frontend_benchmark
from scripts.text_similarity import maximum_similarities, normalize_prompt

OUTPUT = ROOT / "evaluation" / "athlete_router_training_prompts_v2_100.json"
INDEPENDENT = ROOT / "evaluation" / "frontend_realistic_benchmark_60.json"
NEAR_DUPLICATE_LIMIT = 0.82

DOMAINS = (
    "muscular strength", "lower-body dynamics", "muscle-power genetics",
    "blood micronutrients", "basic cognitive function", "mental health",
    "social support", "training conditions",
)

PROMPTS = {
"simple": [
 ("descriptive_statistics","single_cohort_mean","Report the mean standardized social-support score for female athletes, together with the cohort size."),
 ("descriptive_statistics","single_domain_variability","Summarize the mean, standard deviation, and range of training-conditions scores for all athletes."),
 ("descriptive_statistics","filtered_domain_summary","For athletes under 20, report aggregate descriptive statistics for basic cognitive function."),
 ("descriptive_statistics","sport_cohort_summary","Give the median and interquartile range of muscular-strength scores among volleyball athletes."),
 ("descriptive_statistics","two_measure_summary","Summarize average mental-health and social-support standardized scores for senior national-team athletes."),
 ("descriptive_statistics","distribution_summary","Describe the aggregate distribution of lower-body-dynamics scores without exposing individual records."),
 ("descriptive_statistics","sex_filtered_summary","For male athletes, report the mean and variance of blood-micronutrient standardized scores."),
 ("descriptive_statistics","elite_cohort_summary","Return a concise aggregate summary of training conditions for elite athletes."),
 ("descriptive_statistics","count_and_mean","How many semi-elite athletes are in the selected cohort, and what is their average muscular-strength score?"),
 ("descriptive_statistics","domain_pair_summary","Report means and standard deviations for mental health and basic cognitive function in the full cohort."),
 ("correlation_analysis","single_pair_correlation","Calculate the Pearson correlation between muscular strength and lower-body dynamics for all athletes."),
 ("correlation_analysis","filtered_pair_correlation","For female athletes, estimate the correlation between social support and mental health."),
 ("correlation_analysis","sport_pair_association","Measure the association between training conditions and muscular strength among ice-hockey athletes."),
 ("correlation_analysis","three_domain_matrix","Produce a small correlation matrix for mental health, social support, and training conditions."),
 ("correlation_analysis","method_comparison","Report both Pearson and Spearman correlations between lower-body dynamics and muscular strength."),
 ("correlation_analysis","age_filtered_pair","Among athletes aged 20 or above, calculate the correlation between basic cognitive function and social support."),
 ("correlation_analysis","interpret_one_correlation","Calculate the correlation between blood micronutrients and training conditions, then explain its sign in one sentence."),
 ("correlation_analysis","elite_pair_correlation","For elite athletes only, report the Pearson correlation between muscular strength and mental health."),
 ("variance_analysis","one_domain_variance","Run the predefined variance analysis for basic cognitive function across expertise groups."),
 ("variance_analysis","sex_variance_comparison","Compare aggregate variance in lower-body dynamics between male and female athletes."),
 ("variance_analysis","sport_dispersion","Assess whether muscular-strength dispersion differs between volleyball and table-tennis athletes."),
 ("variance_analysis","age_group_variability","Compare training-conditions variability for athletes under 20 versus those aged 20 and above."),
 ("variance_analysis","elite_status_variance","Report the variance comparison for social-support scores by elite status."),
 ("variance_analysis","single_filtered_variance","Estimate mental-health score variance for junior national-team athletes."),
 ("cohort_comparison","two_group_mean","Compare average training-conditions scores between athletes under 20 and athletes aged 20 or above."),
 ("cohort_comparison","sex_mean_difference","Compare mean social-support scores for male and female athletes and report the difference."),
 ("cohort_comparison","expertise_group_difference","Contrast average muscular strength between elite and semi-elite athletes."),
 ("cohort_comparison","sport_group_difference","Compare mean lower-body dynamics for volleyball and ice-hockey cohorts."),
 ("cohort_comparison","team_level_summary","Report the aggregate mental-health difference between junior and senior national-team athletes."),
 ("cohort_comparison","two_domain_group_comparison","Compare muscular strength and training conditions between elite and non-elite groups."),
 ("statistical_interpretation","coefficient_sign","Explain what a positive regression coefficient means in an athlete expertise analysis."),
 ("statistical_interpretation","confidence_interval","Explain how to interpret a confidence interval that includes zero for a standardized-domain coefficient."),
 ("statistical_interpretation","correlation_magnitude","Interpret a correlation of 0.35 between social support and mental health without making a causal claim."),
 ("statistical_interpretation","odds_ratio","In plain language, explain an odds ratio above one in the higher-expertise logistic model."),
 ("statistical_interpretation","variance_result","Explain what greater between-group variance means for training-conditions scores."),
 ("multiple_linear_regression","one_predictor_linear","Fit a simple linear model of expertise value using muscular strength and report the coefficient."),
 ("multiple_linear_regression","two_predictor_linear","Model expertise value from lower-body dynamics and training conditions with no demographic controls."),
 ("logistic_regression","one_predictor_logistic","Run a binary higher-expertise logistic regression using social support as the predictor."),
 ("logistic_regression","two_predictor_logistic","Estimate a logistic model for elite status from muscular strength and lower-body dynamics."),
 ("table1_analysis","single_table1_model","Generate the unadjusted Table 1 logistic model using all eight standardized domains."),
],
"medium": [
 ("descriptive_statistics","stratified_summary","Report mean and standard deviation for four public domains, stratified by sex, with cohort sizes for each stratum."),
 ("descriptive_statistics","multi_cohort_summary","Summarize muscular strength, social support, and training conditions for junior and senior national-team cohorts."),
 ("correlation_analysis","sex_stratified_matrix","Compare correlations among mental health, social support, and training conditions separately for male and female athletes."),
 ("correlation_analysis","filtered_full_matrix","For athletes aged 20 and above, create the eight-domain correlation matrix and identify the three largest absolute correlations."),
 ("correlation_analysis","sport_matrix_comparison","Calculate four-domain correlation matrices for volleyball and ice-hockey athletes and note their largest difference."),
 ("correlation_analysis","partial_interpretation","Examine associations among muscular strength, lower-body dynamics, and training conditions for elite athletes and interpret the pattern."),
 ("variance_analysis","multi_domain_variance","Compare expertise-group variance across muscular strength, mental health, and social support, reporting each result."),
 ("variance_analysis","filtered_variance_panel","For female athletes, run variance comparisons across elite status for all eight standardized domains."),
 ("variance_analysis","variance_with_summary","Assess training-conditions variance by national-team level and accompany it with group means and sample sizes."),
 ("multiple_linear_regression","one_control_linear","For athletes aged 20 and above, regress expertise value on all eight domains while controlling for sex."),
 ("multiple_linear_regression","selected_predictors_linear","Model expertise value using muscular strength, basic cognitive function, mental health, and training conditions; report coefficients and confidence intervals."),
 ("multiple_linear_regression","sex_specific_linear","Fit the same four-domain expertise regression separately for male and female athletes and compare coefficient signs."),
 ("multiple_linear_regression","sport_filtered_linear","For volleyball athletes, estimate the eight-domain expertise model and summarize model fit plus the strongest predictor."),
 ("multiple_linear_regression","controlled_interpretation","Regress expertise value on social support and mental health with age as a control, then interpret both domain coefficients."),
 ("logistic_regression","one_control_logistic","Run the higher-expertise logistic model with all eight domains and sex as the only control."),
 ("logistic_regression","filtered_logistic","For athletes under 20, predict elite status from muscular strength, lower-body dynamics, and training conditions."),
 ("logistic_regression","confidence_interval_logistic","Estimate the binary expertise model from four selected domains and report odds ratios with confidence intervals."),
 ("logistic_regression","sport_logistic","For ice-hockey athletes, fit a logistic model of higher expertise using all standardized domains."),
 ("logistic_regression","two_control_logistic","Model elite status from mental health, social support, and training conditions while controlling for age and sex."),
 ("cohort_comparison","multi_domain_sex_comparison","Compare male and female athletes across muscular strength, mental health, social support, and training conditions."),
 ("cohort_comparison","three_expertise_groups","Compare three expertise groups on lower-body dynamics and basic cognitive function, including group sizes and mean differences."),
 ("cohort_comparison","age_by_sex_comparison","Within each sex, compare average training conditions for under-20 and 20-plus athletes."),
 ("cohort_comparison","sport_pair_multivariate","Contrast volleyball and table-tennis athletes across four standardized domains and highlight the largest gap."),
 ("table1_analysis","table1_subset","Reproduce the sex-adjusted Table 1 model for athletes aged 20 and above and report odds ratios."),
 ("table1_analysis","table1_two_models","Generate unadjusted and age-adjusted Table 1 models, then compare the domain estimates."),
 ("table1_analysis","table1_filtered","Run the Table 1 specification for female athletes using all eight public domains."),
 ("table2_analysis","table2_subset","Reproduce the Table 2 linear model for senior national-team athletes with sex as a control."),
 ("table2_analysis","table2_two_models","Fit unadjusted and sex-adjusted Table 2 models and summarize changes in coefficients."),
 ("table2_analysis","table2_selected_domains","Run a Table 2-style model using four named domains for elite athletes and report model fit."),
 ("statistical_interpretation","several_coefficients","Interpret three standardized regression coefficients, distinguishing statistical direction from practical importance."),
],
"hard": [
 ("correlation_analysis","full_network_reasoning","Analyze correlations among all eight domains, rank the strongest positive and negative relationships, and discuss multiplicity limitations."),
 ("correlation_analysis","cohort_network_comparison","Build separate eight-domain association summaries for elite and semi-elite athletes, then identify relationships that materially differ."),
 ("variance_analysis","variance_sensitivity","Compare variance across expertise groups for all domains, flag unstable estimates, and explain how unequal group sizes affect interpretation."),
 ("multiple_linear_regression","full_vs_filtered_linear","Fit the full-cohort eight-domain expertise model and the 20-plus model, then identify coefficients that change sign or magnitude."),
 ("multiple_linear_regression","nested_linear_models","Fit nested expertise regressions with domains first, then sex, then age; compare fit and coefficient stability across specifications."),
 ("multiple_linear_regression","diagnostic_linear","Estimate the eight-domain expertise regression, report diagnostics for collinearity and residual behavior, and qualify unreliable effects."),
 ("multiple_linear_regression","interaction_reasoning","Compare sex-stratified expertise regressions and assess which apparent coefficient differences warrant an interaction follow-up."),
 ("multiple_linear_regression","multi_cohort_linear","Fit comparable domain models for junior, senior, and non-national-team athletes and synthesize differences without causal claims."),
 ("logistic_regression","nested_logistic_models","Fit higher-expertise logistic models with no controls, sex, age, and both controls; compare odds-ratio stability."),
 ("logistic_regression","calibration_diagnostics","Estimate the full logistic preference model, report discrimination and calibration diagnostics, and explain limitations from cohort size."),
 ("logistic_regression","filtered_model_comparison","Compare eight-domain elite-status models for under-20 and 20-plus athletes, including uncertainty and sparse-outcome cautions."),
 ("logistic_regression","predictor_set_sensitivity","Fit three logistic specifications using physical, psychosocial, and combined domain sets and compare their aggregate performance."),
 ("logistic_regression","robust_interpretation","Estimate a controlled higher-expertise model and explain how correlated predictors complicate interpretation of individual odds ratios."),
 ("cohort_comparison","multi_factor_cohort","Compare elite versus semi-elite athletes across all domains within age strata and summarize consistent versus subgroup-specific gaps."),
 ("cohort_comparison","three_sport_synthesis","Compare volleyball, ice hockey, and table tennis across physical and psychosocial domains, including uncertainty and sample-size caveats."),
 ("table1_analysis","four_table1_models","Generate the four Table 1 logistic models using all eight domains: no controls, sex, age, and both sex and age; compare effect changes."),
 ("table1_analysis","table1_age_sensitivity","Reproduce all Table 1 specifications for the 20-plus cohort and contrast them with full-cohort estimates."),
 ("table1_analysis","table1_sex_stratified","Run the complete Table 1 model sequence separately by sex and discuss unstable or divergent domain effects."),
 ("table1_analysis","table1_domain_blocks","Compare Table 1 models using physical domains, psychosocial domains, and all domains, with the same demographic controls."),
 ("table1_analysis","table1_limitations","Reproduce the fully adjusted Table 1 model, then assess coefficient uncertainty, collinearity, and limits on causal interpretation."),
 ("table1_analysis","table1_sport_comparison","Fit the fully adjusted Table 1 setup for two sport cohorts and compare effect direction while noting small-cell limitations."),
 ("table2_analysis","four_table2_models","Generate four Table 2 linear models with no controls, sex, age, and both controls; compare domain coefficient stability."),
 ("table2_analysis","table2_age_comparison","Fit the eight-domain Table 2 model in the full cohort and age-defined subgroups, then synthesize material changes."),
 ("table2_analysis","table2_diagnostics","Run the fully adjusted Table 2 analysis and report residual, leverage, and multicollinearity diagnostics."),
 ("table2_analysis","table2_domain_blocks","Compare physical-only, psychosocial-only, and combined Table 2 models using consistent age and sex controls."),
 ("table2_analysis","table2_sport_models","Estimate comparable Table 2 models for volleyball, ice hockey, and table tennis and explain cross-cohort uncertainty."),
 ("table2_analysis","table2_sex_models","Run fully adjusted Table 2 models separately for male and female athletes and compare the most influential domains."),
 ("table2_analysis","table2_model_selection","Evaluate several prespecified Table 2 models, compare adjusted fit, and recommend the most defensible specification without data-driven fishing."),
 ("statistical_interpretation","conflicting_models","Explain how to interpret a domain that is positive unadjusted but negative after age and sex controls, including plausible noncausal reasons."),
 ("statistical_interpretation","limitations_synthesis","Given mixed correlation, logistic, and linear-regression findings, write a cautious synthesis separating association, prediction, and causation."),
],
}

REASONS = {
    "simple": "One direct aggregate task with limited filtering or interpretation.",
    "medium": "Multiple conditions, outputs, domains, or a controlled model are required.",
    "hard": "Several related models, diagnostics, subgroup comparisons, or nuanced interpretation are required.",
}


def build_samples() -> list[dict]:
    samples = []
    for difficulty in ("simple", "medium", "hard"):
        for analysis_type, family, prompt in PROMPTS[difficulty]:
            samples.append({
                "id": f"router_v2_{len(samples) + 1:03d}",
                "prompt": prompt,
                "analysis_type": analysis_type,
                "difficulty": difficulty,
                "prompt_family": family,
                "expected_complexity_reason": REASONS[difficulty],
            })
    return samples


def generate(output_path: Path = OUTPUT, independent_path: Path = INDEPENDENT) -> dict:
    samples = build_samples()
    if len(samples) != 100 or len({normalize_prompt(row["prompt"]) for row in samples}) != 100:
        raise RuntimeError("The v2 dataset must contain exactly 100 unique normalized prompts")
    difficulty = Counter(row["difficulty"] for row in samples)
    if difficulty != Counter({"simple": 40, "medium": 30, "hard": 30}):
        raise RuntimeError(f"Unexpected difficulty distribution: {dict(difficulty)}")
    _, independent = load_frontend_benchmark(independent_path)
    evaluation_prompts = [row["prompt"] for row in independent]
    exact = sum(normalize_prompt(row["prompt"]) in {normalize_prompt(p) for p in evaluation_prompts}
                for row in samples)
    similarities = maximum_similarities([row["prompt"] for row in samples], evaluation_prompts)
    near = [(samples[i]["id"], score) for i, score in enumerate(similarities)
            if score >= NEAR_DUPLICATE_LIMIT]
    print(f"Independent evaluation prompts checked: {len(evaluation_prompts)}")
    print(f"Exact overlaps: {exact}")
    print(f"Near-duplicate overlaps: {len(near)}")
    if exact or near:
        raise RuntimeError(f"Training/evaluation overlap detected: exact={exact}, near={near}")
    payload = {
        "dataset_name": "athlete_router_training_prompts_v2_100",
        "purpose": "project_specific_strong_weak_preference_training",
        "independent_evaluation_excluded": "evaluation/frontend_realistic_benchmark_60.json#llm_router_eligible=true",
        "near_duplicate_similarity_limit": NEAR_DUPLICATE_LIMIT,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Difficulty distribution: {dict(difficulty)}")
    print(f"Analysis distribution: {dict(Counter(row['analysis_type'] for row in samples))}")
    return payload


if __name__ == "__main__":
    generate()
