from scripts.validate_dashboard_analysis_matrix import (
    build_cohort_selections,
    build_matrix_cases,
    run_dashboard_matrix,
    validate_frontend_filter_wiring,
)


def test_complete_dashboard_analysis_matrix_without_external_models():
    assert len(build_cohort_selections()) == 17
    assert len(build_matrix_cases()) == 153
    assert validate_frontend_filter_wiring() == []

    rows = run_dashboard_matrix()
    assert len(rows) == 153
    for row in rows:
        assert row["structure_validation"] is True
        assert row["request_match"] is True
        if row["status"] == "NOT_APPLICABLE_STATISTICALLY":
            assert "Small" + "GroupError" not in row["error"]
            assert "MIN_" + "GROUP_SIZE" not in row["error"]
            assert "minimum release " + "size" not in row["error"].lower()
            assert "KeyError" not in row["error"]
            assert "filter" not in row["error"].lower()
            assert any(
                reason in row["error"]
                for reason in (
                    "both elite and semi-elite outcome classes are required",
                    "none of the four model specifications converged",
                    "requires at least two athletes per group",
                )
            )
            assert row["local_execution"] is False
            continue
        assert row["status"] == "PASS", row
        assert row["local_execution"] is True
        assert row["result_validation"] is True
        assert row["filter_match"] is True
        assert row["cohort_size_match"] is True
        assert row["semantic_result_check"] is True
        assert row["noise_utility_check"] is True
