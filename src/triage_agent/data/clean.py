"""Utilitaires de nettoyage de texte.

Appliqués à toutes les sources lors de la standardisation :

- suppression des balises HTML résiduelles ;
- décodage des entités HTML (``&nbsp;``, ``&amp;``…) ;
- suppression optionnelle des marqueurs de citation ``[1]`` ``[2]`` (présents
  dans les réponses MedQuAD / MediQA) ;
- normalisation des espaces (retours à la ligne, tabulations, espaces multiples).
"""

from __future__ import annotations

import html
import re

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[\d+\]")


def clean_text(text: str | None, *, strip_citations: bool = False) -> str:
    """Nettoie un texte et renvoie une chaîne normalisée (jamais ``None``)."""
    if text is None:
        return ""
    s = str(text)
    s = _HTML_RE.sub(" ", s)
    s = html.unescape(s)
    if strip_citations:
        s = _CITATION_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()
