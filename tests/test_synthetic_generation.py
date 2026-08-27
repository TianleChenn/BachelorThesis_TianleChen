from pathlib import Path

import pandas as pd

from data.generate_synthetic_athlete_data import DOMAINS,generate,largest_remainder,validate


ROOT = Path(__file__).resolve().parents[1]

def test_generation_is_reproducible_and_method_derived():
    first,_=generate(2024,300);second,_=generate(2024,300)
    validate(first,300)
    assert first.equals(second)
    assert len(first)==300 and first.athlete_id.nunique()==300
    assert int(first["elite_status"].sum())==25
    assert int((first["expertise_value"]>=13).sum())==25
    assert int((first["elite_status"]==0).sum())==275
    assert list(first[DOMAINS].columns)==DOMAINS
    assert first.elite_status.equals((first.expertise_value>=13).astype(int))


def _assert_stored_dataset_matches_fixed_seed(path: Path) -> None:
    generated,_=generate(2024,300)
    stored=pd.read_csv(path).reset_index(drop=True)
    generated=generated[list(stored.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        stored,
        generated,
        check_dtype=False,
        check_exact=False,
        atol=1e-6,
        rtol=1e-6,
    )
    assert len(generated)==300
    assert int(generated["elite_status"].sum())==25
    assert int((generated["expertise_value"]>=13).sum())==25
    assert int((generated["elite_status"]==0).sum())==275


def test_stored_analysis_dataset_matches_fixed_seed_generator():
    _assert_stored_dataset_matches_fixed_seed(
        ROOT/"data"/"synthetic_athlete_data.csv"
    )


def test_stored_raw_dataset_matches_fixed_seed_generator():
    _assert_stored_dataset_matches_fixed_seed(
        ROOT/"data"/"synthetic_raw_athlete_data.csv"
    )

def test_published_sport_sex_structure_has_exact_total():
    allocation=largest_remainder(300)
    assert sum(count for _,_,count in allocation)==300
    assert not any(sport in {"artistic gymnastics","rhythmic gymnastics"} and sex=="male" for sport,sex,_ in allocation)
