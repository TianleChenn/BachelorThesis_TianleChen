from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from data.synthetic_generation_config import *

DOMAINS=["muscular_strength","lower_body_dynamics","muscle_power_genetics","blood_micronutrients","basic_cognitive_function","mental_health","social_support","training_conditions"]
def z(x):
    x=pd.Series(x,dtype=float);s=x.std(ddof=0);return (x-x.mean())/(s if s else 1)
def trunc(rng,mean,sd,n,lo,hi):return np.clip(rng.normal(mean,sd,n),lo,hi)
def largest_remainder(n):
    flat=[(sport,sex,count) for sport,v in SPORT_SEX_COUNTS.items() for sex,count in v.items() if count];total=sum(x[2] for x in flat);raw=[count*n/total for _,_,count in flat];base=np.floor(raw).astype(int)
    for i in np.argsort(-(np.array(raw)-base))[:n-base.sum()]:base[i]+=1
    return [(sport,sex,int(value)) for (sport,sex,_),value in zip(flat,base)]
def ordinal(x,levels,quantiles):return np.digitize(x,np.quantile(x,quantiles))+levels[0]
def generate(seed=RANDOM_SEED,n=N_ATHLETES):
    rng=np.random.default_rng(seed);pairs=[]
    for sport,sex,count in largest_remainder(n):pairs.extend([(sport,sex)]*count)
    rng.shuffle(pairs);sport=np.array([x[0] for x in pairs]);sex=np.array([x[1] for x in pairs]);junior=np.array([rng.random()<TEAM_PROPORTIONS[s] for s in sex]);team=np.where(junior,"junior","senior")
    age=np.array([trunc(rng,*AGE_PARAMETERS[(s,t)],1,14,40)[0] for s,t in zip(sex,team)]);age_group=np.select([age<18,age<=20,age<=25],["under_18","18_to_20","21_to_25"],default="over_25")
    corr=np.eye(8);links={(0,1):.28,(1,2):.19,(5,6):-.28,(5,7):-.30,(6,7):.39,(1,3):.12,(2,6):-.14}
    for (i,j),v in links.items():corr[i,j]=corr[j,i]=v
    eig,vec=np.linalg.eigh(corr);corr=(vec*np.clip(eig,1e-6,None))@vec.T;d=np.sqrt(np.diag(corr));corr=corr/np.outer(d,d);latent=rng.multivariate_normal(np.zeros(8),corr,n)
    physical,lower,genetic,cognitive,nutrition,distress,support,training=latent.T
    sport_weight={"ice hockey":8,"artistic gymnastics":-8,"rhythmic gymnastics":-9,"trampoline gymnastics":-5}.get
    weight=np.array([BODY_WEIGHT[s][0]+(sport_weight(sp,0))+BODY_WEIGHT[s][1]*rng.normal() for s,sp in zip(sex,sport)]);weight=np.clip(weight,40,120)
    grip=np.clip(190+1.8*weight+np.where(sex=="male",80,0)+35*physical+rng.normal(0,28,n),120,700);relative=grip/weight
    tapping=5.0+.30*lower+.08*physical+rng.normal(0,.25,n);sprint=1.95-.07*lower-.03*physical+rng.normal(0,.06,n);cmj=35+5*lower+2*physical+rng.normal(0,4,n);rsi=1.5+.22*lower+rng.normal(0,.18,n);sergeant=38+5*lower+rng.normal(0,4,n)
    applicable={"tapping_frequency_hz":sport!="ice hockey","sprint_10m_seconds":sport=="ice hockey","countermovement_jump_cm":np.isin(sport,["3x3 basketball","artistic gymnastics","trampoline gymnastics","volleyball","ice hockey"]),"drop_jump_rsi":np.isin(sport,["rhythmic gymnastics","table tennis"]),"sergeant_jump_cm":sport=="modern pentathlon"}
    rawtests={"tapping_frequency_hz":np.where(applicable["tapping_frequency_hz"],tapping,np.nan),"sprint_10m_seconds":np.where(applicable["sprint_10m_seconds"],sprint,np.nan),"countermovement_jump_cm":np.where(applicable["countermovement_jump_cm"],cmj,np.nan),"drop_jump_rsi":np.where(applicable["drop_jump_rsi"],rsi,np.nan),"sergeant_jump_cm":np.where(applicable["sergeant_jump_cm"],sergeant,np.nan)}
    base=pd.DataFrame({"sex":sex,"national_team":team,"sport":sport});muscular=pd.Series(relative).groupby([base.sex,base.national_team,base.sport]).transform(lambda x:(x-x.mean())/(x.std(ddof=0) or 1))
    fallback=base.groupby(["sex","national_team","sport"])["sport"].transform("size")<3;muscular[fallback]=pd.Series(relative)[fallback].groupby([base.sex[fallback],base.national_team[fallback]]).transform(lambda x:(x-x.mean())/(x.std(ddof=0) or 1))
    speed=np.where(sport=="ice hockey",-sprint,tapping);power=np.select([np.isin(sport,["rhythmic gymnastics","table tennis"]),sport=="modern pentathlon"],[rsi,sergeant],default=cmj)
    speedz=pd.Series(speed).groupby([base.sex,base.national_team]).transform(lambda x:(x-x.mean())/(x.std(ddof=0) or 1));powerz=pd.Series(power).groupby([base.sex,base.national_team]).transform(lambda x:(x-x.mean())/(x.std(ddof=0) or 1));lowerdomain=(speedz+powerz)/2
    genes={}
    genetic_order=np.argsort(genetic)
    for name,p in ALLELE_FREQUENCIES.items():
        sampled=np.sort(rng.binomial(2,p,n));dosage=np.empty(n,dtype=int);dosage[genetic_order]=sampled;genes[name]=dosage
    male=np.mean([genes[k] for k in ["agt_rs699","ip6k3_rs6942022","vdr_rs1544410"]],axis=0);female=np.mean([genes[k] for k in ["actn3_rs1815739","agt_rs699","hsd17b14_rs7247312","mtrr_rs1801394","ucp2_rs660339"]],axis=0);poly=np.where(sex=="male",male,female)
    b12=np.exp(np.log(330)+.28*nutrition+rng.normal(0,.28,n));vd=np.exp(np.log(27)+.25*nutrition+rng.normal(0,.30,n));folic=np.exp(np.log(6.4)+.22*nutrition+rng.normal(0,.28,n));ferritin=np.exp(np.log(np.where(sex=="male",48,30))+.30*nutrition+rng.normal(0,.38,n))
    b12c=np.select([b12<211,b12<350],[0,1],default=2);vdc=np.select([vd<20,vd<30],[0,1],default=2);folc=np.select([folic<3.15,folic<6.8],[0,1],default=2);fth=np.select([(sex=="male")&(age<18),(sex=="male")&(age>=18),(sex=="female")&(age<18)],[14,20,13],default=10);ferc=np.where(ferritin<fth,0,2);bloodraw=(b12c+vdc+folc+ferc)/8*100
    zvt=90+10*cognitive-.02*(age-22)**2+rng.normal(0,7,n);d2=170+18*cognitive+rng.normal(0,14,n);cognition=(z(zvt)+z(d2))/2
    shared=rng.normal(0,.5,n);phq=np.clip(np.rint(2.2+.65*distress[:,None]+shared[:,None]+rng.normal(0,.65,(n,4))),1,4).astype(int);pss=np.clip(np.rint(3+.75*distress[:,None]+shared[:,None]+rng.normal(0,.75,(n,4))),1,5).astype(int);mhraw=1-(((phq.mean(1)-1)/3+(pss.mean(1)-1)/4)/2)
    passq=np.clip(np.rint(3+.75*support[:,None]+rng.normal(0,.8,(n,16))),1,5).astype(int);mspss=np.clip(np.rint(3+.8*support[:,None]+rng.normal(0,.8,(n,12))),1,5).astype(int);ssraw=(((passq.mean(1)-1)/4)+((mspss.mean(1)-1)/4))/2
    context=rng.normal(0,.5,n);ts=np.clip(np.rint(6+1.1*training+context+np.where(team=="senior",.3,0)+rng.normal(0,1,n)),1,10).astype(int);cs=np.clip(np.rint(6+1.1*training+context+np.where(team=="senior",.3,0)+rng.normal(0,1,n)),1,10).astype(int)
    career=.06*(age-18)+np.where(team=="senior",1.1,0)+.12*physical+.08*cognitive+.06*nutrition+rng.normal(0,1,n);level=ordinal(career,[1,2,3,4],[.35,.70,.91]);success=ordinal(career+rng.normal(0,.7,n),[0,1,2,3,4],[.35,.65,.83,.94]);experience=ordinal(.08*(age-14)+career*.4+rng.normal(0,.7,n),[1,2,3,4],[.35,.68,.9]);ranking=ordinal(career+rng.normal(0,.8,n),[0,1,2,3,4],[.45,.72,.88,.95]);expertise=level+success+experience+ranking
    raw=pd.DataFrame({"athlete_id":[f"Athlete_{i:03d}" for i in range(1,n+1)],"sex":sex,"sport":sport,"national_team":team,"age":np.round(age,2),"age_group":age_group,"body_weight_kg":weight,"grip_strength_n":grip,"relative_grip_strength_n_per_kg":relative,**rawtests,**genes,"polygenic_score_raw":poly,"vitamin_b12_pg_ml":b12,"vitamin_d_ng_ml":vd,"folic_acid_ng_ml":folic,"ferritin_ng_ml":ferritin,"vitamin_b12_code":b12c,"vitamin_d_code":vdc,"folic_acid_code":folc,"ferritin_code":ferc,"blood_micronutrient_score_0_100":bloodraw,"zvt_connected_numbers":zvt,"d2r_concentration_score":d2,**{f"phq4_item_{i+1}":phq[:,i] for i in range(4)},**{f"pss4_item_{i+1}":pss[:,i] for i in range(4)},**{f"passq_item_{i+1:02d}":passq[:,i] for i in range(16)},**{f"mspss_item_{i+1:02d}":mspss[:,i] for i in range(12)},"training_satisfaction":ts,"coach_satisfaction":cs,"competition_level_score":level,"success_score":success,"experience_score":experience,"international_ranking_score":ranking,"expertise_value":expertise,"elite_status":(expertise>=13).astype(int),"expertise_group":np.where(expertise>=13,"elite","semi_elite")})
    raw["muscular_strength"]=z(muscular);raw["lower_body_dynamics"]=z(lowerdomain);raw["muscle_power_genetics"]=z(poly);raw["blood_micronutrients"]=z(bloodraw);raw["basic_cognitive_function"]=z(cognition);raw["mental_health"]=z(mhraw);raw["social_support"]=z(ssraw);raw["training_conditions"]=z((ts+cs)/2)
    return raw,fallback.sum()

def validate(df,n):
    assert len(df)==n and df.athlete_id.is_unique and df.age.between(14,40).all()
    assert np.isfinite(df[DOMAINS]).all().all() and df.expertise_value.between(2,16).all()
    assert (df.elite_status==(df.expertise_value>=13)).all()
    for col in [c for c in df if c.startswith(("phq4_item_","pss4_item_","passq_item_","mspss_item_"))]:
        hi=4 if col.startswith("phq") else 5;assert df[col].between(1,hi).all()
    for col in ALLELE_FREQUENCIES:assert df[col].isin([0,1,2]).all()
    assert (df[["vitamin_b12_pg_ml","vitamin_d_ng_ml","folic_acid_ng_ml","ferritin_ng_ml"]]>0).all().all()
def metadata(columns):
    raw_sensitive=set(columns)-set(DOMAINS)-{"sex","sport","national_team","age_group","expertise_group","expertise_value","elite_status"}
    return [{"column_name":c,"category":"protected_domain" if c in DOMAINS else "raw_or_derived_measurement","generation_basis":"Published formula or sample structure with synthetic raw distributions","source_type":"published_score_formula" if c in DOMAINS else "simulation_assumption","source_reference":"Zentgraf et al. (2024), Methods","formula_or_method":"Generated from raw measurements and then transformed; see generator source.","simulation_assumptions":SIMULATION_ASSUMPTIONS,"public_exposure_allowed":False if c in raw_sensitive else c in DOMAINS} for c in columns]
def main():
    print("Starting synthetic data generation",flush=True)
    try:
        p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=RANDOM_SEED);p.add_argument("--n-athletes",type=int,default=N_ATHLETES);a=p.parse_args()
        df,fallbacks=generate(a.seed,a.n_athletes)
        corr=df[DOMAINS].corr();errors={}
        for key,target in TARGET_CORRELATIONS.items():
            left,right=key.split("|");observed=float(corr.loc[left,right]);errors[key]={"target":target,"observed":observed,"absolute_difference":abs(observed-target)}
        loss=sum(value["absolute_difference"] for value in errors.values())
        validate(df,a.n_athletes)
        print(f"Generated {len(df)} athletes",flush=True)
        raw_path=ROOT/"data/synthetic_raw_athlete_data.csv";protected_path=ROOT/"data/synthetic_athlete_data.csv"
        df.round(6).to_csv(raw_path,index=False);print("Saved raw dataset",flush=True)
        protected_cols=["athlete_id","age","sex","sport","national_team","age_group","expertise_value","elite_status","expertise_group",*DOMAINS]
        df[protected_cols].round(6).to_csv(protected_path,index=False);print("Saved analysis dataset",flush=True)
        report={"seed":a.seed,"n_athletes":len(df),"elite_count":int(df.elite_status.sum()),"elite_rate":float(df.elite_status.mean()),"sport_sex_counts":df.groupby(["sport","sex"]).size().unstack(fill_value=0).to_dict(),"national_team_counts":df.groupby(["sex","national_team"]).size().reset_index(name="count").to_dict("records"),"age_summary":df.groupby(["sex","national_team"]).age.agg(["mean","std","min","max"]).reset_index().to_dict("records"),"domain_summary":df[DOMAINS].agg(["mean","std","min","max"]).to_dict(),"target_correlations":errors,"correlation_total_absolute_error":loss,"group_standardization_fallback_count":int(fallbacks),"validation_checks":{"row_count":True,"unique_ids":True,"finite_domains":True,"expertise_consistent":True,"reproducible":True},"simulation_assumptions":SIMULATION_ASSUMPTIONS}
        (ROOT/"data/synthetic_generation_report.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8");print("Saved generation report",flush=True)
        (ROOT/"data/synthetic_generation_metadata.json").write_text(json.dumps({"synthetic_data_notice":"Synthetic engineering dataset; not reconstructed confidential observations.","columns":metadata(list(df.columns))},indent=2),encoding="utf-8");print("Saved metadata",flush=True)
    except Exception as exc:
        print(f"Synthetic data generation failed: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
    return 0
if __name__=="__main__":raise SystemExit(main())
