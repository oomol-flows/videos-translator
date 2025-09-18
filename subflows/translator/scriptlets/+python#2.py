def main(params: dict):
  segments: list[dict[str, str]] = params["segments"]
  texts: list[str] = [s["text"] for s in segments]
  return { "texts": texts }
