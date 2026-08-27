import pytest
from ui.data_generation_helpers import calculate_expertise_summary,format_expertise_range
def test_expertise_summary_and_bins():
    values=[2,4,5,8,9,12,13,16]
    result=calculate_expertise_summary(values)
    assert result["mean"]==8.625 and result["median"]==8.5
    assert result["min"]==2 and result["max"]==16
    assert result["higher_expertise_count"]==2
    assert list(result["distribution_bins"])==["2–4","5–8","9–12","13–16"]
    assert sum(result["distribution_bins"].values())==len(values)
    assert format_expertise_range(2,16)=="2–16"
def test_empty_and_invalid_expertise():
    assert calculate_expertise_summary([])=={}
    with pytest.raises(ValueError):calculate_expertise_summary([1,17])
    with pytest.raises(ValueError):calculate_expertise_summary([2,None])
