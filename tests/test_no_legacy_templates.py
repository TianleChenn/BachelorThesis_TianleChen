from pathlib import Path
def test_generation_and_verifier_have_no_legacy_templates():
    generator=Path("llm/code_generator.py").read_text(encoding="utf-8")
    verifier=Path("llm/generated_code_verifier.py").read_text(encoding="utf-8")
    for text in ["result = run_table1()","result = run_table2(","result = run_figure1()","result = run_figure2(","result = run_variance()","result = run_network("]:
        assert text not in generator and text not in verifier
    assert "ALLOWED_FUNCTIONS" not in verifier
    assert not Path("llm/code_executor.py").exists()
