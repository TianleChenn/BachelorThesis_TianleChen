from privacy.evaluation_metrics import ROUTES,calculate_privacy_metrics
def row(actual,predicted):return {"ground_truth_route":actual,"predicted_route":predicted,"correct":actual==predicted}
def test_perfect_privacy_metrics():
    result=calculate_privacy_metrics([row(r,r) for r in ROUTES]);assert result["overall_accuracy"]==1;assert result["macro_f1"]==1
def test_privacy_failure_rates_and_matrix():
    result=calculate_privacy_metrics([row("collaboration","cloud"),row("local_edge","cloud"),row("blocked","cloud"),row("cloud","blocked")])
    assert result["privacy_safety_metrics"]["sensitive_to_cloud_rate"]==1
    assert result["privacy_safety_metrics"]["blocked_bypass_rate"]==1
    assert result["privacy_safety_metrics"]["overprotection_rate"]==.25

def test_privacy_safety_matrix_cases():
    rows=[row("collaboration","local_edge"),row("local_edge","collaboration"),
          row("cloud","collaboration"),row("blocked","local_edge"),
          row("blocked","blocked")]
    calculate_privacy_metrics(rows)
    assert rows[0]["safety_score"]==1 and not rows[0]["strict_correct"] and rows[0]["safety_aware_correct"]
    assert rows[1]["safety_score"]==0 and not rows[1]["safety_aware_correct"]
    assert rows[2]["safety_score"]==.9 and not rows[2]["strict_correct"] and rows[2]["safety_aware_correct"]
    assert rows[3]["safety_score"]==.25 and not rows[3]["safety_aware_correct"]
    assert rows[4]["safety_score"]==1 and rows[4]["safety_aware_correct"]

def test_privacy_metrics_are_bounded_and_empty_input_is_safe():
    for rows in ([],[row(a,p) for a in ROUTES for p in ROUTES]):
        result=calculate_privacy_metrics(rows)
        values=[result["overall_accuracy"],result["strict_privacy_accuracy"],
                result["privacy_safety_score"],result["safety_aware_privacy_accuracy"],
                result["exact_route_match_rate"],
                *result["privacy_safety_metrics"].values()]
        assert all(0<=value<=1 for value in values)
    assert list(result["confusion_matrix"])==ROUTES

def test_requested_confusion_matrix_protection_order_metrics():
    rows=(
        [row("cloud","cloud") for _ in range(9)]
        +[row("collaboration","collaboration") for _ in range(7)]
        +[row("collaboration","blocked") for _ in range(2)]
        +[row("local_edge","local_edge") for _ in range(4)]
        +[row("local_edge","collaboration") for _ in range(3)]
        +[row("blocked","blocked") for _ in range(5)]
    )
    result=calculate_privacy_metrics(rows);safety=result["privacy_safety_metrics"]
    assert result["exact_route_match_rate"]==25/30
    assert result["safety_aware_privacy_accuracy"]==27/30
    assert safety["underprotection_rate"]==3/30
    assert safety["overprotection_rate"]==2/30

def test_exact_errors_partition_into_under_and_overprotection():
    rows=[row(actual,predicted) for actual in ROUTES for predicted in ROUTES]
    result=calculate_privacy_metrics(rows);safety=result["privacy_safety_metrics"]
    assert result["safety_aware_privacy_accuracy"]>=result["exact_route_match_rate"]
    assert 1-result["exact_route_match_rate"]==safety["underprotection_rate"]+safety["overprotection_rate"]
