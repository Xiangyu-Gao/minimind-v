import re
import time
import asyncio


def normalize(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return set(text.split())


def jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def safe_translate(
    translator_obj,
    text: str,
    src: str = "en",
    dest: str = "zh-cn",
    retries: int = 3,
    delay: float = 1.0,
) -> str:
    """Translate text with retries and fall back to the original text on failure.

    This wraps `translator.translate` and handles both sync and coroutine
    implementations, retrying on exceptions with exponential backoff.
    """
    if not text:
        return text

    for attempt in range(1, retries + 1):
        try:
            res = translator_obj.translate(text, src=src, dest=dest)
            if asyncio.iscoroutine(res):
                # run coroutine safely
                try:
                    res = asyncio.run(res)
                except RuntimeError:
                    # If an event loop is already running (e.g. in some envs),
                    # fall back to calling without awaiting and treat as failure
                    raise

            return res.text if hasattr(res, "text") else str(res)
        except Exception as e:
            print(f"Translation attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))
            else:
                print("Translation failed after retries, returning original text.")
                return text
