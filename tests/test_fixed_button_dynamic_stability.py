from pathlib import Path
from unittest.mock import patch

from llm.generated_code_verifier import inspect_generated_code
from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.code_generation_pools import ANALYSIS_METHOD_POOL
from llm.code_generation_prompt import build_code_generation_messages
from llm.code_generator import _generate
from llm.model_clients import ModelCallResult


def _call(content):
    return ModelCallResult(content, "mock", "mock", "test", True, False, False, None)


def _generate_with(responses, requested_analysis):
    calls = iter(_call(item) for item in responses)
    with patch("llm.code_generator._call_for_channel", side_effect=lambda *_: next(calls)):
        return _generate("strong", "safe aggregate request", {}, "request-1", requested_analysis, "mock")


def test_shared_prompt_exposes_every_method_for_every_request():
    messages=build_code_generation_messages("Calculate Pearson correlation.")
    prompt=repr(messages)
    for method in ANALYSIS_METHOD_POOL:
        assert method in prompt


def test_invalid_cloud_outputs_receive_exactly_one_repair():
    contract = render_request_contract(build_request_contract("figure1"))
    invalid = [
        contract.replace("result = ", "", 1),
        "analysis\n" + contract,
        contract.replace("analysis.figure1", "analysis.network_analysis"),
        contract.replace("variance_iterations=1000", "iterations=1000"),
    ]
    for first in invalid:
        generated, _, error, repaired, stage, _ = _generate_with([first, first], "figure1")
        assert generated is None
        assert error
        assert repaired is True
        assert stage in {"format_validation", "request_validation"}


def test_twenty_repeated_contracts_per_fixed_button_are_stable():
    for method in ANALYSIS_METHOD_POOL:
        contract=render_request_contract(build_request_contract(method))
        for _ in range(20):
            validation = inspect_generated_code(contract,user_request="safe aggregate request",
                requested_analysis=method,requested_filters={})
            assert validation.request_match_passed
            assert validation.generated_method == method


def test_arbitrary_python_is_never_accepted():
    for code in [
        "import os\n" + render_request_contract(build_request_contract("figure1")),
        "result = __import__('os').system('whoami')",
        "while True:\n    pass",
        render_request_contract(build_request_contract("figure1")) + "\nresult = analysis.figure1(variables=[])",
    ]:
        assert not inspect_generated_code(code,user_request="figure1",
            requested_analysis="figure1",requested_filters={}).request_match_passed


def test_frontend_fixed_buttons_pass_explicit_intent():
    source = Path("frontend.py").read_text(encoding="utf-8")
    for method in {"table1","table2","figure1","figure2","correlation","variance_analysis"}:
        assert f'"requested_analysis": "{method}"' in source
    assert '"requested_analysis": "individual_profile"' not in source
    assert "Individual athlete analysis" in source
    assert 'requested_analysis=analysis["requested_analysis"]' in source


def test_removed_dashboard_plot_modules_are_not_available():
    frontend = Path("frontend.py").read_text(encoding="utf-8")
    assert '"title": "Network plot"' not in frontend
    assert '"button": "Generate network plot"' not in frontend
    assert '"title": "Variance plot"' not in frontend
    assert '"button": "Plot variance"' not in frontend
    assert "network_analysis" not in ANALYSIS_METHOD_POOL
