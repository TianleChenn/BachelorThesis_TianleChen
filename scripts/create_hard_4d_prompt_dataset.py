"""Create the manually labelled hard 4D prompt-only dataset (no LLM calls)."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=PROJECT_ROOT/"evaluation/privacy_gating_train_4d_hard_independent_90_prompts.json"

CLOUD=[
("Explain how Table 1 logistic regression is interpreted for athlete expertise without requesting any results.","method_explanation","Sensitive context is mentioned only as methodology; no athlete data are requested."),
("Describe the difference between Table 1 and Table 2 in the Athlete Data Analysis System.","method_explanation","A public explanation of supported aggregate methods has minimal disclosure."),
("Explain how Figure 1 represents conditional relationships among standardized athlete domains.","method_explanation","The request concerns network methodology rather than observed data."),
("Explain why Figure 2 uses standardized z-scores and an elite reference profile.","method_explanation","The request asks about visualization design without generating profiles."),
("Describe how blood micronutrients could be used in an aggregate regression, without calculating any athlete result.","sensitive_explanation","A highly sensitive domain name appears, but only in a methodological explanation."),
("Explain privacy considerations for mental-health analysis in athlete research without accessing the dataset.","sensitive_explanation","This is a privacy and method explanation, not a data request."),
("Describe how muscle-power genetics is represented as a derived standardized domain.","sensitive_explanation","Genetics is discussed at the domain-definition level only."),
("Run Table 1 for the complete athlete cohort using only protected aggregate outputs.","broad_aggregate","A full-cohort protected regression has broad scope and derived output."),
("Run Table 2 for all athletes and return only aggregate coefficients and confidence intervals.","broad_aggregate","The full cohort and aggregate coefficients avoid individual disclosure."),
("Generate Figure 1 for the complete cohort using standardized domain correlations.","broad_aggregate","The network is based on broad aggregate derived values."),
("Generate Figure 2 as an aggregate variance comparison across all eight domains, with no individual traces.","broad_aggregate","Explicitly excludes individual profiles while using the full cohort."),
("Calculate correlations among all standardized domains for the full athlete sample.","broad_aggregate","Broad-scope correlation returns only aggregate coefficients."),
("Compare variance across standardized athlete domains for all sports combined.","broad_aggregate","The complete population makes the aggregate output low exposure."),
("Provide descriptive statistics for all athletes using derived z-scores only.","broad_aggregate","Only broad derived summaries are requested."),
("Summarize the full-cohort distribution of expertise scores without subgroup breakdowns.","broad_aggregate","No small group or individual can be isolated."),
("List dataset metadata, column definitions, and available aggregate analyses.","metadata","Only public structural information is requested."),
("Report the total number of athlete records and the names of the eight derived domains.","metadata","Counts and schema names do not reveal athlete values."),
("Explain how the athlete variables are standardized into z-scores.","standardization","This is a public statistical explanation."),
("Explain why the overall standardized mean equals zero in the athlete dataset.","standardization","No observed athlete output is requested."),
("Describe how missing values are handled before aggregate athlete regression.","method_explanation","Preprocessing methodology alone is low disclosure."),
("Explain odds ratios using a hypothetical full-cohort athlete example.","method_explanation","The example is hypothetical and contains no dataset result."),
("Explain standardized beta coefficients in an athlete Table 2 analysis.","method_explanation","The request is educational and non-personal."),
("Describe network centrality for athlete-domain Figure 1 without computing it.","method_explanation","No data access or result generation occurs."),
("Explain variance inflation factors for the full-cohort athlete regression.","method_explanation","A broad methodological explanation remains cloud-suitable."),
("Run a full-cohort correlation between mental-health and blood-domain z-scores using aggregate results only.","broad_sensitive_aggregate","Although domains are sensitive, the cohort is maximally broad and output aggregate."),
("Compare aggregate genetic-domain and training-condition means across the entire athlete sample.","broad_sensitive_aggregate","Sensitive genetics appears only in a broad protected summary."),
("Generate full-sample descriptive statistics for psychological derived scores without subgroup filters.","broad_sensitive_aggregate","Large scope and derived statistics reduce disclosure despite sensitivity."),
("Explain what an anonymous individual profile means, but do not generate one.","sensitive_explanation","The request names an individual analysis type but asks only for explanation."),
("Describe how the system protects athlete identifiers before analysis.","metadata","This is a public privacy-control explanation."),
("Give a public overview of the Athlete Data Analysis System and its supported figures and tables.","metadata","A product overview contains no protected athlete results.")]

COLLABORATION=[
("Run an aggregate correlation of blood micronutrients for all elite athletes.","sensitive_cohort","Sensitive derived data are requested for a filtered but non-identifiable group."),
("Compare mental-health z-score means between female and male athlete cohorts.","sensitive_cohort","Sensitive psychological results require protected collaboration despite aggregate output."),
("Run Table 2 using muscle-power genetics predictors for senior national-team athletes.","sensitive_cohort","Genetic predictors and a filtered team cohort create moderate privacy risk."),
("Generate Figure 1 linking blood, mental-health, and social-support domains for elite athletes.","sensitive_cohort","Multiple sensitive domains are aggregated within a filtered expertise group."),
("Analyze variance in basic cognitive function for female athletes.","filtered_cohort","A demographic filter and cognitive domain warrant prompt protection."),
("Calculate aggregate biometric correlations for national-team athletes.","sensitive_cohort","Biometric information in a defined cohort is sensitive but not individual."),
("Run logistic regression on mental-health derived scores for under-20 athletes.","sensitive_cohort","Age filtering plus psychological features creates a protected aggregate request."),
("Compare blood-micronutrient means across elite and semi-elite groups.","sensitive_cohort","Sensitive group comparison remains aggregate but benefits from collaboration."),
("Generate Table 2 for female national-team athletes using social-support and training-condition scores.","filtered_cohort","Intersecting cohort descriptors narrow the group without naming an individual."),
("Create an aggregate genetics network for athletes in one sport discipline.","sensitive_cohort","Genetic-domain aggregation within one sport is sensitive and filtered."),
("Summarize mental-health and cognitive z-scores for a protected cohort of twelve athletes.","small_sensitive_cohort","The cohort is small and sensitive, but no identity or individual output is requested."),
("Calculate training-condition statistics for a five-athlete group without identifiers.","small_nonsensitive_cohort","Low-sensitivity data are requested for a small aggregate group, creating a boundary case."),
("Compare public sport categories for a six-athlete cohort using counts only.","small_nonsensitive_cohort","The variables are low sensitivity but the group is small."),
("Run aggregate blood-domain variance for the complete national-team cohort.","broad_sensitive_aggregate","The cohort is large, while the data domain is highly sensitive."),
("Analyze full-cohort genetic correlations by returning only a protected summary.","broad_sensitive_aggregate","Highly sensitive data remain group-level and protected."),
("Generate aggregate mental-health descriptive statistics for all athletes, using local perturbation before cloud analysis.","broad_sensitive_aggregate","Broad scope lowers subject risk, but psychological data justify collaboration."),
("Create Figure 2 profiles representing group means for a filtered elite cohort, with no individual lines.","filtered_cohort","Group profiles are derived but the cohort is filtered."),
("Compare standardized blood and genetic domains for female elite athletes as aggregate means.","sensitive_cohort","Two filters and highly sensitive domains require protected collaboration."),
("Run Table 1 for junior national-team athletes using cognitive and mental-health predictors.","sensitive_cohort","A defined youth cohort and sensitive predictors warrant collaboration."),
("Correlate social support with mental health in semi-elite athletes.","sensitive_cohort","Psychological data in an expertise subgroup are protected aggregate content."),
("Analyze training-condition variance for a small anonymous team subgroup.","small_nonsensitive_cohort","A small group is a privacy boundary even with less-sensitive derived data."),
("Generate a network of standardized domains for a ten-athlete protected cohort.","small_sensitive_cohort","Multiple domains and a small cohort need local protection before cloud work."),
("Compare cognitive-function means between two national-team levels.","filtered_cohort","The group analysis is non-identifiable but concerns a sensitive cognitive domain."),
("Run aggregate regression for blood micronutrients across all female athletes.","sensitive_cohort","A large demographic cohort still uses sensitive blood information."),
("Summarize genetic-domain variance in elite athletes without showing any athlete profile.","sensitive_cohort","The output is aggregate, yet the genetic domain and filtered group remain sensitive."),
("Calculate correlations for an anonymous cohort selected by sport and age group.","filtered_cohort","Combined filters narrow the cohort even without identity."),
("Compare mental-health domain distributions across sport disciplines using protected aggregates.","broad_sensitive_aggregate","The analysis is broad but sensitive and benefits from collaboration safeguards."),
("Create Table 2 for a seven-athlete cohort using only standardized training scores.","small_nonsensitive_cohort","Derived training values are less sensitive, while seven subjects form a small group."),
("Generate aggregate blood-micronutrient statistics for a large cohort and suppress all individual values.","broad_sensitive_aggregate","This explicit large-cohort sensitive request is a boundary between cloud and collaboration."),
("Analyze standardized psychological and biometric outcomes for a filtered cohort with no athlete IDs.","sensitive_cohort","No identity is present, but sensitive data and filtering require protected collaboration.")]

LOCAL=[
("Generate an anonymous standardized individual profile for one randomly selected athlete.","anonymous_individual","One person's derived profile remains individual even without a public identifier."),
("Compare one anonymous athlete's derived profile with the elite mean.","individual_vs_group","An individual-to-group comparison can disclose a unique protected pattern."),
("Show Athlete_23's protected eight-domain z-score profile without raw measurements.","identifiable_individual","A direct athlete ID makes the derived request identifiable."),
("Interpret Athlete_91's exact derived expertise score locally.","identifiable_individual","The value is derived rather than raw, but it is exact and tied to an ID."),
("Analyze Athlete_265's standardized blood-domain profile on the local edge.","identifiable_individual","Identifiable sensitive blood results must remain local."),
("Summarize one identifiable athlete's public sport category and competition level.","low_sensitive_individual","The attributes are comparatively low sensitivity, but the subject is identifiable."),
("Return one athlete's derived training-condition scores without raw questionnaire answers.","anonymous_individual","No raw values are requested, yet the output remains a detailed individual result."),
("Create a protected mental-health profile for one anonymous athlete.","anonymous_individual","An anonymous subject still has a sensitive individual psychological profile."),
("Generate Athlete_23's standardized muscle-power genetics profile locally.","identifiable_individual","Genetic-domain output associated with an athlete ID is high risk."),
("Compare Athlete_91's mental-health z-scores with national-team averages.","individual_vs_group","The comparison exposes identifiable sensitive deviation from a group."),
("Interpret one athlete's blood micronutrient z-scores against the full-cohort mean.","individual_vs_group","An individual sensitive profile is compared to a public reference."),
("Generate Figure 2 for one anonymous athlete and the elite reference line.","anonymous_individual","Figure 2 is derived but represents one unique subject."),
("Create an individual network of standardized domains for Athlete_265.","identifiable_individual","A detailed derived network is linked to an athlete ID."),
("Run a local-only correlation analysis of one athlete's repeated protected scores.","anonymous_individual","The analysis targets one subject and must not leave the edge."),
("Analyze variance for a uniquely selected athlete's derived measurements locally.","anonymous_individual","Unique selection creates individual scope despite anonymized display."),
("Compare one female national-team athlete's protected profile with all female athletes.","individual_vs_group","A narrowly described individual is contrasted with her cohort."),
("Provide the precise standardized result for Athlete_23's cognitive domain.","identifiable_individual","An exact derived cognitive result is tied to a direct identifier."),
("Interpret one anonymous athlete's social-support and training-condition profile.","anonymous_individual","Detailed multidomain information is about one person."),
("Generate a local profile for one semi-elite athlete selected from a small team.","anonymous_individual","Selection context may make an ostensibly anonymous athlete identifiable."),
("Analyze one national-team athlete's public team level and derived expertise profile.","low_sensitive_individual","Some fields are public-like, but the combined individual profile remains local."),
("Compare two athletes' standardized training-condition profiles without displaying their IDs.","very_small_cohort","A two-person cohort is too small for remote aggregate processing."),
("Run descriptive statistics for a uniquely filtered three-athlete cohort locally.","very_small_cohort","Three subjects create strong singling-out risk."),
("Calculate protected correlations for four athletes from one national-team subgroup.","very_small_cohort","The very small filtered cohort requires local processing."),
("Generate a detailed derived profile for one athlete but omit all raw data.","anonymous_individual","Absence of raw data does not eliminate individual-profile risk."),
("Return Athlete_91's exact standardized blood and mental-health results.","identifiable_individual","Exact derived sensitive values are linked to an ID."),
("Analyze an identifiable athlete's genetic and biometric profile locally only.","identifiable_individual","Multiple highly sensitive domains and identity require local edge."),
("Compare one anonymous athlete with a five-athlete reference group.","individual_vs_group","The target remains an individual and the reference group is also small."),
("Create a protected local Table 2 interpretation for Athlete_265's derived outcome.","identifiable_individual","A regression-style interpretation is still a person-specific result."),
("Show one athlete's exact derived profile score for each of the eight domains.","anonymous_individual","Eight exact derived scores form a detailed individual fingerprint."),
("Perform local-only analysis of one athlete's protected cognitive, genetic, and mental-health profile.","anonymous_individual","The request combines sensitive individual domains and explicitly requires local execution.")]

def build():
    samples=[]
    for route,rows in (("cloud",CLOUD),("collaboration",COLLABORATION),("local_edge",LOCAL)):
        for prompt,family,rationale in rows:
            samples.append({"id":f"hard4d_{len(samples)+1:03d}","prompt":prompt,"ground_truth_route":route,"prompt_family":family,"difficulty":"hard","review_status":"pending","rationale":rationale})
    return {"schema_version":"llm-4d-hard-prompts-v1","dataset_name":"privacy_gating_train_4d_hard_independent_90","rules_version":"athlete-privacy-rubric-v8-continuous-no-level","feature_source":"to_be_generated_by_active_llm_privacy_assessor","route_annotation":"manual_project_specific","samples":samples}
def validate(payload):
    rows=payload["samples"]; ids=[r["id"] for r in rows]; prompts=[" ".join(r["prompt"].casefold().split()) for r in rows]; counts=Counter(r["ground_truth_route"] for r in rows); invalid=sum(r["ground_truth_route"] not in {"cloud","collaboration","local_edge"} for r in rows); features=sum("features" in r for r in rows)
    if len(rows)!=90 or counts!={"cloud":30,"collaboration":30,"local_edge":30} or len(ids)!=len(set(ids)) or len(prompts)!=len(set(prompts)) or invalid or features: raise ValueError("Generated hard prompt dataset failed validation")
    return counts,len(ids)-len(set(ids)),len(prompts)-len(set(prompts)),invalid,features
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",default=str(DEFAULT_OUTPUT));parser.add_argument("--overwrite",action="store_true");args=parser.parse_args();path=Path(args.output)
    if path.exists() and not args.overwrite: raise FileExistsError(f"Dataset already exists: {path}. Use --overwrite to replace it.")
    payload=build();counts,duplicate_ids,duplicate_prompts,invalid,features=validate(payload);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Created:\n{path.relative_to(PROJECT_ROOT).as_posix()}\n");print(f"Samples: {len(payload['samples'])}");print(f"Cloud: {counts['cloud']}");print(f"Collaboration: {counts['collaboration']}");print(f"Local Edge: {counts['local_edge']}");print(f"Duplicate IDs: {duplicate_ids}");print(f"Duplicate Prompts: {duplicate_prompts}");print(f"Invalid Routes: {invalid}");print(f"Features Present: {features}");return 0
if __name__=="__main__":raise SystemExit(main())
