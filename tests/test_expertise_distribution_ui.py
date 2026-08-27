from pathlib import Path
def test_continuous_expertise_presentation_and_internal_grouping():
    frontend=Path("frontend.py").read_text(encoding="utf-8")
    assert '"Mean Expertise Score"' in frontend
    assert '"Expertise Score Range"' in frontend
    assert '"Random Seed"' not in frontend
    assert '"Elite Athletes"' not in frontend
    assert '"Semi-Elite Athletes"' not in frontend
    assert "Expertise Distribution" in frontend
    assert "Higher-Expertise Group for Logistic Regression" in frontend
    analysis=Path("sports/analysis.py").read_text(encoding="utf-8")
    assert 'df["elite_status"]' in analysis
