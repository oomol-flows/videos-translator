def main(params: dict):
  texts: list[str] = params["texts"]
  segments: list[dict] = params["segments"]
  for text, segment in zip(texts, segments):
    segment["text"] = text
  return { "segments": segments }
