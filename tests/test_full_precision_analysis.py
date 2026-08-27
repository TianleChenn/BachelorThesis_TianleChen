from sports.analysis import fit_table2_models_raw, load_data, run_table2
from sports.config import PREDICTORS


def test_table2_public_and_raw_results_keep_full_precision():
    dataframe = load_data()
    raw = fit_table2_models_raw(dataframe, group="all")
    public = run_table2(dataframe=dataframe)
    raw_beta = raw["models"][0]["coefficients"][PREDICTORS[0]]["coefficient"]
    public_beta = next(row for row in public["rows"] if row["variable"] != "Intercept")["beta_nature"]
    assert public_beta == raw_beta
    assert public_beta != round(public_beta, 2)
    assert public["model_stats"][0]["r_squared"] == raw["models"][0]["r_squared"]
