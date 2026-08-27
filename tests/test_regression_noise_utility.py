from sports.analysis import load_data, run_table2
from sports.config import PREDICTORS
from sports.regression_noise_utility import evaluate_table2_noise_utility


def _legend_labels(figure):
    return {
        label
        for axis in figure.axes
        for label in axis.get_legend_handles_labels()[1]
        if label
    }


def test_table2_noise_utility_has_fifty_runs_and_all_models():
    result = evaluate_table2_noise_utility(load_data())
    assert result["repetitions"] == 50
    assert result["noise_range"] == [-0.50, 0.50]
    assert result["experiment_version"] == "uniform-0.50-all-analyses-v2"
    assert result["perturbed_columns"] == PREDICTORS
    assert result["independent_repetitions"] is True
    assert result["cumulative_noise"] is False
    assert result["outcome_perturbed"] is False
    assert len(result["model_summary"]) == 4
    assert len(result["coefficient_summary"]) == 8
    assert len(result["primary_summary"]["coefficient_rmse_per_run"]) == 50
    assert result["primary_model"].startswith("d)")
    assert hasattr(result["stability_figure"], "savefig")
    metrics=[result["primary_summary"][key] for key in ["baseline_r_squared","mean_noisy_r_squared","mean_delta_r_squared","mean_coefficient_rmse"]]
    assert all(isinstance(value,float) for value in metrics)
    assert any(value != round(value,4) for value in metrics)
    perturbed = result["mean_perturbed_table2"]
    assert perturbed["repetitions"] == 50
    assert perturbed["rows"]
    required = {"group", "model", "variable", "beta_nature", "standardized_beta", "SE", "t", "p", "robust_standard_error_type"}
    assert required.issubset(perturbed["rows"][0])
    baseline_rows = run_table2(dataframe=load_data())["rows"]
    baseline_keys = {(row["group"], row["model"], row["variable"]) for row in baseline_rows}
    perturbed_keys = {(row["group"], row["model"], row["variable"]) for row in perturbed["rows"]}
    assert perturbed_keys == baseline_keys
    assert len(perturbed["rows"]) == len(baseline_rows)
    assert "athlete_id" not in repr(perturbed).lower()
    assert "noise_matrix" not in repr(perturbed).lower()

    mean_result = result["mean_perturbed_result"]
    assert mean_result["title"] == "Mean Perturbed Table 2-style Linear Regression"
    assert mean_result["table"] == perturbed["rows"]
    assert not hasattr(mean_result.get("figure"), "savefig")

    difference = result["average_difference"]
    assert difference["label"] == "Average Standardized Coefficient Difference"
    assert difference["value"] == result["primary_summary"]["mean_coefficient_rmse"]
    assert difference["explanation"]

    legend_labels = _legend_labels(result["stability_figure"])
    assert "Original" in legend_labels
    assert "Mean perturbed" in legend_labels
    assert any("1 SD" in label for label in legend_labels)
