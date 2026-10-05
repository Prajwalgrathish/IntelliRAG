"""Prompt templates enforcing grounded, cited answers."""
from __future__ import annotations

from typing import Optional

NO_CONTEXT_ANSWER = "I could not find enough information in the provided documents to answer this question."

SYSTEM_PROMPT = f"""You are a careful document question-answering assistant.
You answer using ONLY the numbered context passages supplied in the user message.

Rules:
1. Use only information that appears in the context. Do not use outside knowledge.
2. If the context does not contain enough information, reply with exactly: "{NO_CONTEXT_ANSWER}" You may then briefly mention what related information the context does contain.
3. Cite supporting passages inline with their numbers, for example [1] or [2][3]. Only cite numbers that exist in the context. Never invent sources, titles, quotes or page numbers.
4. Clearly separate what the context states from anything uncertain, ambiguous or only partially supported, and say when the context is incomplete.
5. Be concise. Do not repeat the question."""

INTENT_GUIDANCE: dict[str, str] = {
    "summary": "Provide a concise, structured summary of the relevant context.",
    "comparison": "Compare the items explicitly, noting similarities and differences found in the context.",
    "procedural": "Present the steps in order, as stated in the context.",
    "definition": "Give a precise definition using the wording and meaning found in the context.",
    "factual": "Give the specific fact requested, quoting figures or names exactly as in the context.",
}

USER_TEMPLATE = """Context:
{context}

{hint}Question: {question}

Answer (use inline [n] citations, and only information from the context):"""


def build_user_prompt(question: str, context: str, intent: Optional[str] = None) -> str:
    """Fill the user prompt template, adding intent-specific guidance when available."""
    guidance = INTENT_GUIDANCE.get(intent or "")
    hint = f"Question type: {intent}. {guidance}\n\n" if guidance else ""
    return USER_TEMPLATE.format(context=context, hint=hint, question=question)


def is_abstention(answer: str) -> bool:
    """True if the answer states that the context is insufficient."""
    normalised = " ".join(answer.lower().split())
    return normalised.startswith(NO_CONTEXT_ANSWER.lower()[:45]) or NO_CONTEXT_ANSWER.lower() in normalised
