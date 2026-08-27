from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

DOMAINS=["Muscular Strength","Lower-Body Dynamics","Muscle-Power Genetics","Blood Micronutrients","Basic Cognitive Function","Mental Health","Social Support","Training Conditions"]
def load_generation_report(path):
    try:return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError,json.JSONDecodeError):return None
def load_generation_metadata(path):
    payload=load_generation_report(path) or {};rows=payload.get("columns") or []
    allowed={"column_name","category","source_type","source_reference","formula_or_method","simulation_assumptions","public_exposure_allowed"}
    return [{k:row.get(k) for k in allowed} for row in rows if isinstance(row,dict)]
def load_safe_dataset_summary(csv_path):
    path=Path(csv_path)
    if not path.exists():return {}
    try:
        df=pd.read_csv(path,usecols=lambda c:c in {"sex","sport","national_team","age","elite_status","expertise_group","expertise_value"})
        return {"number_of_athletes":len(df),"number_of_sports":int(df.sport.nunique()) if "sport" in df else None,
          "sex_counts":df.sex.value_counts().to_dict() if "sex" in df else {},"sport_counts":df.sport.value_counts().to_dict() if "sport" in df else {},"team_counts":df.national_team.value_counts().to_dict() if "national_team" in df else {},"elite_counts":df.expertise_group.value_counts().to_dict() if "expertise_group" in df else df.elite_status.value_counts().rename(index={0:"semi_elite",1:"elite"}).to_dict() if "elite_status" in df else {},"age_histogram":df.age.value_counts(bins=8,sort=False).rename_axis("Age Range").reset_index(name="Athletes").astype({"Age Range":str}).to_dict("records") if "age" in df else [],"expertise_summary":calculate_expertise_summary(df.expertise_value) if "expertise_value" in df else {}}
    except Exception:return {}
def build_domain_construction_rows():
    data=[("Muscular Strength","Body weight, absolute grip strength","Relative grip strength; z-standardized by sex, national team, and sport","Published method"),("Lower-Body Dynamics","Sport-specific tapping, sprint, countermovement jump, drop jump, or sergeant jump tests","Relevant tests separately standardized by sex and national team, then combined","Published method"),("Muscle-Power Genetics","Sex-specific SNP dosage values","Sex-specific polygenic score, then whole-sample z-standardization","Published gene structure; simulated allele frequencies"),("Blood Micronutrients","Vitamin B12, vitamin D, folic acid, ferritin","Clinical threshold coding, 0-to-100 score, then z-standardization","Published thresholds and formula; simulated raw distributions"),("Basic Cognitive Function","ZVT and d2-R test scores","Separate z-standardization, then combination","Published method"),("Mental Health","PHQ-4 and PSS-4 questionnaire items","Scale means, common range transformation, direction alignment, then z-standardization","Published scales and aggregation; direction alignment is an implementation choice"),("Social Support","PASS-Q and MSPSS questionnaire items","Scale means, common range transformation, average, then z-standardization","Published method"),("Training Conditions","Training satisfaction and coaching-staff satisfaction","Mean of both items, then z-standardization","Published method")]
    return [{"Domain":a,"Raw Inputs":b,"Processing":c,"Scientific Basis":d,"Raw Data Visible to LLM":"No","Raw Data Visible to Frontend":"No"} for a,b,c,d in data]
def build_visibility_matrix():
    return [{"Data Layer":a,"Trusted Backend":b,"Cloud LLM":c,"Local Edge Model":d,"Frontend":e} for a,b,c,d,e in [("Raw diagnostic measurements","Yes","No","Only when strictly required locally","No"),("Full athlete rows","Yes","No","No","No"),("Raw identifiers","Yes","No","Local-only when required","No"),("Eight domain schema","Yes","Yes","Yes","Yes"),("Aggregate statistics","Yes","After PRISM routing","Yes","Yes"),("Individual standardized profile","Yes","No real identifier","Local only","Pseudonymized"),("Generation metadata","Yes","Safe summary only","Yes","Yes")]]
def normalize_correlation_report(report):
    rows=[]
    for relationship,value in (report or {}).get("target_correlations",{}).items():
        if not isinstance(value,dict):continue
        rows.append({"Relationship":relationship.replace("|"," ↔ ").replace("_"," ").title(),"Published Target":value.get("target"),"Synthetic Observed":value.get("observed"),"Absolute Difference":value.get("absolute_difference"),"Within Tolerance":value.get("absolute_difference") is not None and value["absolute_difference"]<=.12})
    return rows
def validate_no_row_level_exposure(data):
    if isinstance(data,list):return not any(isinstance(x,dict) and "athlete_id" in x for x in data)
    return not isinstance(data,pd.DataFrame)

def calculate_expertise_summary(expertise_values):
    values=pd.to_numeric(pd.Series(expertise_values),errors="coerce")
    if values.isna().any():raise ValueError("Expertise values must be numeric and non-missing.")
    if values.empty:return {}
    if not values.between(2,16).all():raise ValueError("Expertise values must be between 2 and 16.")
    bins=build_expertise_distribution_rows(values)
    return {"mean":float(values.mean()),"median":float(values.median()),"std":float(values.std(ddof=1)),"min":float(values.min()),"max":float(values.max()),"higher_expertise_count":int((values>=13).sum()),"comparison_group_count":int((values<=12).sum()),"distribution_bins":{row["Expertise Score Range"]:row["Number of Athletes"] for row in bins}}

def build_expertise_distribution_rows(expertise_values):
    values=pd.to_numeric(pd.Series(expertise_values),errors="coerce")
    if values.isna().any():raise ValueError("Expertise values must be numeric and non-missing.")
    ranges=[("2–4",2,4),("5–8",5,8),("9–12",9,12),("13–16",13,16)]
    return [{"Expertise Score Range":label,"Number of Athletes":int(values.between(low,high,inclusive="both").sum())} for label,low,high in ranges]

def format_expertise_range(minimum,maximum):
    if minimum is None or maximum is None:return "Not available"
    return f"{minimum:g}–{maximum:g}"

def get_expertise_group_explanation():
    return ("Following the reference study, logistic regression internally converts the continuous expertise score into two analysis groups:\n\n"
            "Higher-expertise group: expertise score >= 13\n\nComparison group: expertise score <= 12\n\n"
            "This grouping is used only for statistical analysis and should not be interpreted as an absolute judgment of an athlete's ability.")
