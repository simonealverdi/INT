# interviewer_reflection.py
from __future__ import annotations
import os, time
from typing import List

from dotenv import load_dotenv
from openai import OpenAI                    # ⬅ nuovo client

load_dotenv()
client = OpenAI()                            # usa OPENAI_API_KEY

# ----------------------------------------------------------
class InterviewerReflection:
    """Mantiene il transcript e genera le ‘riflessioni’ (summary + topic-tracking)."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.transcript: List[dict] = []     # [{"speaker":"user","text":...}, …]
        self.reflections: List[str] = []     # testo sintetico periodico
        self._chars_since_last = 0

    # ---------- API pubblica ----------
    def add_turn(self, speaker: str, text: str) -> None:
        self.transcript.append({"speaker": speaker, "text": text.strip()})
        self._chars_since_last += len(text)
        # se supero ~200 caratteri di nuovo materiale → rifletto
        if self._chars_since_last >= 200:
            self.reflect()

    def get_context(self, k_last: int = 6) -> str:
        """Restituisce una view compatta (ultimi k turni + ultime riflessioni)."""
        last_turns = self.transcript[-k_last:]
        turns_txt = "\n".join(f"[{t['speaker']}] {t['text']}" for t in last_turns)
        refl_txt  = "\n".join(self.reflections[-3:])
        return refl_txt + "\n" + turns_txt

    # ---------- internals ----------
    def reflect(self) -> None:
        """Invia transcript ≈ ridotto al modello e salva la sintesi."""
        if not self.transcript:
            return

        prompt_sys = (
            "Sei un assistente che legge il transcript di un’intervista "
            "e produce riflessioni concise (max 6 bullet)."
        )
        prompt_usr = (
            "\n".join(f"[{t['speaker']}] {t['text']}" for t in self.transcript[-10:])
            + "\n\n### TASK\nSintetizza gli elementi nuovi o importanti in bullet-point."
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user",   "content": prompt_usr},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        summary = resp.choices[0].message.content.strip()
        self.reflections.append(f"**Reflection {len(self.reflections)+1}:**\n{summary}")
        self._chars_since_last = 0
