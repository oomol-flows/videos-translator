from oocana import Context
from shared.translation import Translater
from .llm_parser import parse

def main(params: dict, context: Context):
  llm = parse(params, context)
  timeout: float | None = params["timeout"]
  if timeout == 0.0:
    timeout = None

  translater = Translater(
    api=llm.api,
    key=llm.key,
    url=llm.url,
    model=llm.model,
    temperature=params["temperature"],
    timeout=timeout,
    source_lan=params["source"],
    target_lan=params["target"],
    group_max_tokens=params["max_translating_group"],
    streaming=True,
  )
  target_texts = translater.translate(
    source_texts=params["texts"],
    report_progress=lambda p: context.report_progress(p * 100.0),
  )
  return { "texts": target_texts }