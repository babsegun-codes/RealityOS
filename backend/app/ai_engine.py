import json
import os
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """You are the intelligence engine for RealityOS.
Analyze a user's document and convert unstructured information into useful, conservative, actionable intelligence.
Never invent facts. Only infer when the document strongly supports the inference, and mark uncertain items in the reasoning.
Return JSON only.

Schema:
{
  "summary": "short factual summary",
  "document_type": "contract|bill|invoice|letter|job_offer|admission|government_notice|insurance|other",
  "entities": [{"type":"person|organization|location|other","name":"..."}],
  "facts": [{"label":"...","value":"..."}],
  "obligations": [{"title":"action the user is required or strongly expected to take","description":"why it matters","due_date":"YYYY-MM-DD or null","priority":"low|medium|high|critical","consequence":"likely consequence stated or clearly implied by the document, or null"}],
  "deadlines": [{"date":"YYYY-MM-DD or null","description":"what the date refers to"}],
  "amounts": [{"value":"exact amount as written","description":"what the amount refers to"}],
  "attention_items": [{"title":"...","reason":"...","urgency":"low|medium|high|critical"}]
}
Use null for dates that cannot be determined. Do not create an obligation merely because a document mentions an event. Dates without a clear year should use the supplied reference date to resolve the year only when reasonable.
"""


def analyze_document(text: str, reference_date: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    user_prompt = f"Reference date: {reference_date}\n\nDOCUMENT TEXT:\n{text[:60000]}"

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AI returned invalid JSON") from exc
