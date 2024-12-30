from oocana import Context
from shared.translation import Translater, LLM_API

def main(params: dict, context: Context):
  api: LLM_API
  llm_api: str = params["llm_api"]
  if llm_api == "openai":
    api = LLM_API.OpenAI
  elif llm_api == "claude":
    api = LLM_API.Claude
  elif llm_api == "gemini":
    api = LLM_API.Gemini
  else:
    raise ValueError(f"Invalid LLM API: {llm_api}")

  translater = Translater(
    api=api,
    key=params["key"],
    url=params["url"],
    model=params["model"],
    temperature=params["temperature"],
    timeout=params["timeout"],
    source_lan=params["source_lan"],
    target_lan=params["target_lan"],
    group_max_tokens=params["group_max_tokens"],
  )
  target_texts = translater.translate(
    source_texts=params["texts"],
    report_progress=lambda p: context.report_progress(p * 100.0),
  )
  return { "texts": target_texts }
