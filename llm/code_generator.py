"""Fresh LLM generation of one strictly restricted analysis API call."""
from __future__ import annotations
import json,re,uuid
import logging
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from .code_generation_prompt import CODE_GENERATION_PROMPT_VERSION,RESTRICTED_CODE_MAX_TOKENS,build_code_generation_messages
from .generated_code_verifier import extract_restricted_assignment,inspect_generated_code
from .env import load_local_env
from .model_clients import call_gemini_cloud_model,call_local_codegen_model
from sports.restricted_analysis_api import ALLOWED_METHODS
from privacy.athlete_id import redact_athlete_ids

load_local_env()
logger = logging.getLogger(__name__)
@dataclass
class CodeGenerationResult:
    code:str|None;code_source:str;action:str;group:str|None;explanation:str
    schema_only_context_sent:bool=True;raw_data_sent_to_llm:bool=False;cloud_prompt_source:str="none"
    privacy_applied_to_cloud_prompt:bool=False;requested_generator_channel:str="dynamic_restricted_code_generator"
    used_generator_channel:str="none";generator_fallback_used:bool=False;direct_answer_generated:bool=False
    requested_model:str|None=None;actual_model:str|None=None;provider:str|None=None;model_call_success:bool=False
    model_unavailable:bool=False;cloud_semantic_sketch_received:bool=False;local_plan_validated:bool=False
    original_prompt_sent_to_cloud:bool=False;local_edge_endpoint_enforced:bool=False
    generated_code_type:str="restricted_python";structure_validation_attempted:bool=False
    validation_error:str|None=None;generation_retry_used:bool=False;provider_retry_used:bool=False
    restricted_api_used:bool=True
    generation_timestamp:str="";generation_request_id:str="";fixed_template_generator_used:bool=False
    failure_stage:str|None=None;requested_analysis:str|None=None
    normalization_applied:bool=False;structure_validation_passed:bool=False
    request_match_passed:bool=False
    generator_target:str|None=None;cloud_used:bool=False
    local_generator_available:bool|None=None;first_generation_non_empty:bool=False
    first_validation_passed:bool=False;repair_validation_passed:bool=False
    first_validation_error:str|None=None;repair_validation_error:str|None=None
    candidate_assignment_count:int=0
    prompt_version:str=CODE_GENERATION_PROMPT_VERSION
def to_dict(obj):return asdict(obj) if hasattr(obj,"__dataclass_fields__") else dict(obj)
def detect_requested_analysis(prompt):
    lower=str(prompt or "").lower()
    return ("figure1" if "figure 1" in lower else "figure2" if "figure 2" in lower else
      "table1" if "table 1" in lower or "logistic regression models" in lower else
      "table2" if "table 2" in lower else
      "variance_analysis" if "variance" in lower else "correlation" if "correlation" in lower else "unspecified")

def is_unsupported_future_prediction(prompt:str)->bool:
    text=" ".join(str(prompt or "").lower().split())
    patterns=[
        r"\bwill\b.*\bbecome\b.*\belite\b",
        r"\bpredict whether\b.*\b(?:will )?become elite\b",
        r"\bfuture elite status\b",
        r"\bfuture performance classification\b",
        r"\bforecast elite status\b",
        r"\bpredict promotion to elite\b",
    ]
    return any(re.search(pattern,text) for pattern in patterns)

def _call_for_channel(channel,messages):
    if channel=="local":return call_local_codegen_model(messages,max_tokens=RESTRICTED_CODE_MAX_TOKENS)
    if channel=="cloud":return call_gemini_cloud_model(messages,max_tokens=RESTRICTED_CODE_MAX_TOKENS)
    raise ValueError(f"Unsupported code-generation tier: {channel}")
def _model_unavailable_stage(channel:str)->str:
    if channel=="local":return "local_model_unavailable"
    if channel=="cloud":return "cloud_model_unavailable"
    return "code_generation"
def _audit(entry):
    logger.debug("dynamic_generation_audit %s", json.dumps(entry, ensure_ascii=False))
def _local_audit(entry):
    logger.debug("local_generation_debug %s", json.dumps(entry, ensure_ascii=False))
def _response_preview(value,private=False):
    if private:return "[REDACTED PRIVATE MODEL RESPONSE]"
    return redact_athlete_ids(str(value or ""))[:500]
def _generate(channel,prompt,context,request_id,requested_analysis,selected_model,figure2_size=None,
              validation_prompt=None):
    requested_filters=dict(context.get("requested_filters") or {})
    base_messages=build_code_generation_messages(prompt)
    metadata={"local_generator_available":None,"first_generation_non_empty":False,
      "first_validation_passed":False,"repair_validation_passed":False,
      "first_validation_error":None,"repair_validation_error":None,"candidate_assignment_count":0,
      "provider_retry_used":False}
    previous_code="";previous_raw="";error=None;call=None;raw="";validation_stage="format_validation"
    attempts=2;provider_retry_remaining=1
    for attempt in range(attempts):
        messages=list(base_messages)
        if attempt:
            safe_error=redact_athlete_ids(str(error or "Unknown local validation error."))
            retry_message=(
              "The previous restricted call failed local validation.\n"
              f"Validation error: {safe_error}\n"
              "Correct only the invalid syntax or arguments and return exactly one "
              "result = analysis.<method>(...) assignment. Return code only."
            )
            messages.extend([
              {"role":"assistant","content":previous_raw},
              {"role":"user","content":retry_message},
            ])
        call=_call_for_channel(channel,messages)
        if call.unavailable and provider_retry_remaining:
            provider_retry_remaining-=1
            metadata["provider_retry_used"]=True
            call=_call_for_channel(channel,messages)
        if channel=="local":metadata["local_generator_available"]=not call.unavailable
        if not call.success or not call.content:
            error=call.error or "Selected model returned no restricted code."
            stage=_model_unavailable_stage(channel) if call.unavailable else "code_generation"
            _audit({"generation_request_id":request_id,"requested_analysis":requested_analysis,"selected_model":selected_model,
              "generated_method":None,"validation_stage":stage,"sanitized_validation_error":error,
              "raw_response_length":len(str(call.content or "")),"raw_response_preview":_response_preview(call.content,channel=="local"),
              "cleaned_code_preview":"","candidate_assignment_count":0,"first_validation_error":None,
              "repair_attempted":attempt>0,"repair_validation_error":None})
            return None,call,error,attempt>0,stage,metadata
        if attempt==0:metadata["first_generation_non_empty"]=True
        raw=str(call.content);previous_raw=raw;previous_code,candidate_count=extract_restricted_assignment(raw)
        metadata["candidate_assignment_count"]=candidate_count
        validation=inspect_generated_code(previous_code,user_request=validation_prompt or prompt,
          requested_analysis=requested_analysis,requested_filters=requested_filters)
        if validation.request_match_passed:
            method,args=validation.generated_method,validation.generated_arguments
            if attempt==0:metadata["first_validation_passed"]=True
            else:metadata["repair_validation_passed"]=True
            if channel=="local":_local_audit({"requested_analysis":requested_analysis,"model_id":selected_model,
              "parsed_method":method,"structure_validation":"PASS","request_match":"PASS",
              "repair_attempted":attempt>0,"sanitized_validation_error":None})
            _audit({"generation_request_id":request_id,"requested_analysis":requested_analysis,"selected_model":selected_model,
              "generated_method":method,"validation_stage":"request_validation","sanitized_validation_error":None,
              "raw_response_length":len(raw),"raw_response_preview":_response_preview(raw,channel=="local"),
              "cleaned_code_preview":_response_preview(validation.cleaned_code,channel=="local"),
              "candidate_assignment_count":candidate_count,"first_validation_error":metadata["first_validation_error"],
              "repair_attempted":attempt>0,"repair_validation_error":None})
            return (validation.cleaned_code,method,args),call,None,attempt>0,raw.strip()!=validation.cleaned_code,metadata
        error=validation.validation_error;validation_stage=validation.failure_stage or "format_validation"
        if attempt==0:metadata["first_validation_error"]=error
        else:metadata["repair_validation_error"]=error
        if attempt==1:
            _audit({"generation_request_id":request_id,"requested_analysis":requested_analysis,"selected_model":selected_model,
              "generated_method":validation.generated_method,"validation_stage":validation.failure_stage,"sanitized_validation_error":error,
              "raw_response_length":len(raw),"raw_response_preview":_response_preview(raw,channel=="local"),
              "cleaned_code_preview":_response_preview(previous_code,channel=="local"),"candidate_assignment_count":candidate_count,
              "first_validation_error":metadata["first_validation_error"],"repair_attempted":True,
              "repair_validation_error":metadata["repair_validation_error"]})
            return None,call,error,attempt>0,validation_stage,metadata
    return None,call,error,True,validation_stage,metadata
def _failure(reason,request_id,timestamp,call=None,retry=False,privacy=False,source="none",failure_stage="code_generation",requested_analysis=None):
    return CodeGenerationResult(None,source,"generation_failed",None,"Restricted code generation failed.",
      privacy_applied_to_cloud_prompt=privacy,used_generator_channel="none",requested_model=getattr(call,"requested_model",None),
      actual_model=getattr(call,"actual_model",None),provider=getattr(call,"provider",None),model_call_success=False,
       model_unavailable=getattr(call,"unavailable",True),structure_validation_attempted=True,validation_error=reason,
      generation_retry_used=retry,generation_timestamp=timestamp,generation_request_id=request_id,failure_stage=failure_stage,requested_analysis=requested_analysis)
def generate_code(prompt,model_decision,privacy_decision,use_openai=True,requested_analysis=None,requested_filters=None):
    request_id=str(uuid.uuid4());timestamp=datetime.now(timezone.utc).isoformat();route=privacy_decision.get("route")
    if is_unsupported_future_prediction(prompt):
        return CodeGenerationResult(None,"unsupported","unsupported",None,
          "Future elite-status prediction is not supported.",requested_generator_channel="none",
          generation_timestamp=timestamp,generation_request_id=request_id,failure_stage="unsupported")
    if privacy_decision.get("blocked") or route=="blocked":return CodeGenerationResult(None,"blocked","blocked",None,"PRISM blocked the request.",requested_generator_channel="none",generation_timestamp=timestamp,generation_request_id=request_id)
    selected=str(model_decision.get("selected_model") or "")
    channel=str(model_decision.get("selected_tier") or "none")
    requested_analysis=requested_analysis or detect_requested_analysis(prompt)
    if requested_analysis not in ALLOWED_METHODS:
        return _failure("Unsupported requested analysis.",request_id,timestamp,source="dynamic_restricted_python",failure_stage="unsupported_analysis",requested_analysis=requested_analysis)
    cloud_sketch=False;privacy_applied=route=="collaboration" and channel=="cloud"
    original_cloud=route=="cloud" and channel=="cloud";context={}
    if route=="local_edge":channel="local";selected="local_ministral";original_cloud=False
    if route=="blocked":return _failure("PRISM blocked the request.",request_id,timestamp,source="blocked")
    if channel not in {"cloud","local"}:
        return _failure("No executable Cloud/Local tier was selected.",request_id,timestamp,source="dynamic_restricted_python")
    generation_prompt=(privacy_decision.get("cloud_prompt") if route=="collaboration" and channel=="cloud" else prompt)
    if not generation_prompt:
        return _failure("No privacy-approved prompt was available for Cloud generation.",request_id,timestamp,
                        source="dynamic_restricted_python")
    if not use_openai and channel!="local":return _failure("Configured LLM generation is disabled.",request_id,timestamp,source="dynamic_restricted_python")
    requested_filters=dict(requested_filters or {})
    context["requested_filters"]=requested_filters
    generation=_generate(channel,generation_prompt,context,request_id,requested_analysis,selected,
                         validation_prompt=prompt)
    generated,call,error,retry=generation[:4];generation_meta=generation[5] if len(generation)>5 else {}
    if not generated:
        failure_stage=generation[4]
        failure=_failure(error,request_id,timestamp,call,retry,privacy_applied,"dynamic_restricted_python",failure_stage,requested_analysis)
        generator_channel=f"{channel}_restricted_code_generator"
        failure.requested_generator_channel=generator_channel
        failure.used_generator_channel=generator_channel
        failure.generator_target=("local_restricted_generator" if channel=="local"
            else "cloud_restricted_generator")
        failure.cloud_used=channel=="cloud"
        if route=="local_edge":
            failure.local_edge_endpoint_enforced=True
            failure.generator_target="local_restricted_generator"
        for key,value in generation_meta.items():setattr(failure,key,value)
        return failure
    code,method,args=generated
    filters=args.get("filters") if isinstance(args.get("filters"),dict) else {};normalized=bool(generation[4])
    return CodeGenerationResult(code,"fresh_llm_restricted_python",method,filters.get("sport"),"Fresh restricted Python generated and AST validated.",
      cloud_prompt_source="two_layer_ldp_perturbed_prompt" if privacy_applied else "original_prompt" if original_cloud else "none",
      privacy_applied_to_cloud_prompt=privacy_applied,requested_generator_channel=f"{channel}_restricted_code_generator",
      used_generator_channel=f"{channel}_restricted_code_generator",requested_model=call.requested_model,
      actual_model=call.actual_model,provider=call.provider,model_call_success=True,model_unavailable=False,cloud_semantic_sketch_received=cloud_sketch,
      local_plan_validated=True,original_prompt_sent_to_cloud=original_cloud,local_edge_endpoint_enforced=channel=="local",structure_validation_attempted=True,
      generation_retry_used=retry,generation_timestamp=timestamp,generation_request_id=request_id,requested_analysis=requested_analysis,
      normalization_applied=normalized,structure_validation_passed=True,request_match_passed=True,
      generator_target="local_restricted_generator" if channel=="local" else "cloud_restricted_generator",
      cloud_used=channel=="cloud",**generation_meta)
