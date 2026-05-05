"""
persona/events.py — Life event detection from message text.

Detects mentions of major life milestones:
    marriage        — "I got married", "my wedding", "my husband/wife"
    birth           — "had a baby", "expecting", "pregnant", "newborn"
    graduation      — "graduated", "finished school", "got my degree"
    job_change      — "got a new job", "quit my job", "was fired", "promoted"
    moving          — "moved to", "moving to", "new apartment", "new house"
    loss            — "passed away", "died", "lost my", "funeral"
    divorce         — "divorced", "separated", "split up", "ex-wife/husband"
    health_event    — "diagnosed", "surgery", "hospital stay", "recovery"

Only first-person events (I/my/we) are captured.

Usage:
    from persona.events import EventDetector
    events = EventDetector().detect(messages)
"""

from __future__ import annotations

import re
from data.models import Message
from persona.schema import PersonaLifeEvent

# (event_type, pattern)
EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("marriage", re.compile(
        r"\b(got married|we got married|my wedding|my husband|my wife|"
        r"i married|we married|tied the knot|engagement|got engaged)\b",
        re.IGNORECASE,
    )),
    ("birth", re.compile(
        r"\b(had a baby|having a baby|we're expecting|i'm pregnant|"
        r"newborn|gave birth|my baby|my infant|new baby|baby shower)\b",
        re.IGNORECASE,
    )),
    ("graduation", re.compile(
        r"\b(graduated|i graduated|finished school|finished college|"
        r"got my degree|got my diploma|commencement|my graduation)\b",
        re.IGNORECASE,
    )),
    ("job_change", re.compile(
        r"\b(got a new job|started a new job|quit my job|left my job|"
        r"i was fired|got fired|laid off|i got promoted|new position|"
        r"switching careers|career change|starting a business)\b",
        re.IGNORECASE,
    )),
    ("moving", re.compile(
        r"\b(i moved|we moved|moving to|just moved|new apartment|new house|"
        r"new home|relocated|relocation|buying a house|bought a house)\b",
        re.IGNORECASE,
    )),
    ("loss", re.compile(
        r"\b(passed away|he died|she died|they died|lost my (mom|dad|"
        r"mother|father|sister|brother|friend|dog|cat|pet|husband|wife)|"
        r"funeral|grieving|in mourning)\b",
        re.IGNORECASE,
    )),
    ("divorce", re.compile(
        r"\b(divorced|getting divorced|we divorced|separated|my ex-wife|"
        r"my ex-husband|my ex|split up|we broke up|ended our marriage)\b",
        re.IGNORECASE,
    )),
    ("health_event", re.compile(
        r"\b(diagnosed with|i was diagnosed|had surgery|going to surgery|"
        r"recovering from|hospital stay|my operation|health scare|"
        r"cancer|heart attack|stroke|hospitalized)\b",
        re.IGNORECASE,
    )),
]

# Only capture if sentence contains first-person reference
FIRST_PERSON = re.compile(r"\b(i|my|we|our|me|mine|myself)\b", re.IGNORECASE)


class EventDetector:
    """Detects first-person life event mentions across a speaker's messages."""

    def detect(self, messages: list[Message]) -> list[PersonaLifeEvent]:
        events: list[PersonaLifeEvent] = []
        seen: set[str] = set()  # Deduplicate by event_type + description

        for msg in messages:
            text = msg.text
            if not FIRST_PERSON.search(text):
                continue  # Skip non-first-person messages

            for event_type, pattern in EVENT_PATTERNS:
                if pattern.search(text):
                    key = f"{event_type}:{text[:50].lower()}"
                    if key not in seen:
                        seen.add(key)
                        events.append(PersonaLifeEvent(
                            event_type=event_type,
                            description=text[:150],
                            conversation_id=msg.global_index,
                            confidence=0.85,
                        ))

        return events
