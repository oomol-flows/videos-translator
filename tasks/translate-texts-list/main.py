from oocana import Context
from shared.translation import Translater
from .llm_parser import parse

#region generated meta
import typing
class LLMModelOptions(typing.TypedDict):
  model: str
  temperature: float
  top_p: float
  max_tokens: int
class Inputs(typing.TypedDict):
  texts: list[str]
  timeout: float | None
  llm: LLMModelOptions
  source: typing.Literal["en", "cn", "ja", "fr", "ru", "de"]
  target: typing.Literal["en", "cn", "ja", "fr", "ru", "de"]
  group_max_tokens: int
class Outputs(typing.TypedDict):
  texts: list[str]
#endregion

def main(params: dict, context: Context):
  llm_model = params["llm"]
  timeout: float | None = params["timeout"]
  if timeout == 0.0:
    timeout = None

  translater = Translater(
    key=context.oomol_llm_env["api_key"],
    url=context.oomol_llm_env["base_url_v1"],
    model=llm_model["model"],
    temperature=llm_model["temperature"],
    timeout=timeout,
    source_lan=params["source"],
    target_lan=params["target"],
    group_max_tokens=params["group_max_tokens"],
    streaming=True,
  )
  target_texts = translater.translate(
    source_texts=params["texts"],
    report_progress=lambda p: context.report_progress(p * 100.0),
  )
  return { "texts": target_texts }