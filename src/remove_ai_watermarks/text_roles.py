"""Optional English loan-role checks over caller-supplied dependency parses.

This module neither loads a model nor establishes semantic equivalence. A true
result means only that the selected parsed loan roles show no known reversal.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any

_NOMINAL = frozenset({"NOUN", "PROPN", "PRON"})
_SUBJECT = frozenset({"nsubj", "nsubjpass", "nsubj:pass"})
_OBJECT = frozenset({"dobj", "obj"})
_CONTROL = frozenset(
    {"happy", "willing", "ready", "eager", "reluctant", "able", "want", "plan", "decide", "try", "hope", "agree"}
)
_UNRESOLVED_PRONOUN = frozenset(
    {"i", "me", "we", "us", "you", "he", "him", "she", "her", "they", "them", "it", "what", "which", "that"}
)
_RoleKey = tuple[bool, str | None, str]


def _nominals(token: Any) -> list[Any]:
    return [token, *(item for item in token.conjuncts if item.pos_ in _NOMINAL)]


def _identity(token: Any) -> str | None:
    lemma = token.lemma_.casefold()
    if token.pos_ not in _NOMINAL or lemma in _UNRESOLVED_PRONOUN:
        return None
    return " ".join([*(item.lemma_.casefold() for item in token.lefts if item.dep_ == "compound"), lemma])


def _subjects(token: Any) -> list[Any]:
    visited: set[int] = set()
    current = token
    while current.i not in visited:
        visited.add(current.i)
        found = [child for child in current.children if child.dep_ in _SUBJECT]
        if found:
            return found
        if (
            current.dep_ == "conj"
            or (current.dep_ == "xcomp" and current.head.lemma_.casefold() in _CONTROL)
            or (current.dep_ == "acomp" and current.lemma_.casefold() in _CONTROL and current.head.lemma_ == "be")
        ):
            current = current.head
        else:
            break
    return []


def _add_roles(roles: dict[_RoleKey, set[str]], negated: bool, items: set[str], entity: Any, role: str) -> None:
    for nominal in _nominals(entity):
        name = _identity(nominal)
        if name is None:
            continue
        roles[(negated, None, name)].add(role)
        for item in items:
            roles[(negated, item, name)].add(role)


def _loan_roles(doc: Any) -> dict[_RoleKey, set[str]]:
    if not len(doc) or not all(doc.has_annotation(name, require_complete=True) for name in ("DEP", "POS", "LEMMA")):
        raise ValueError("Loan-role checks require complete dependency, POS and lemma annotations")
    roles: dict[_RoleKey, set[str]] = defaultdict(set)
    for token in doc:
        verb = token.lemma_.casefold()
        if token.pos_ != "VERB" or verb not in {"borrow", "lend", "loan"}:
            continue
        children = list(token.children)
        subjects = _subjects(token)
        objects = [child for child in children if child.dep_ in _OBJECT]
        # An English recipient passive retains its direct object: residents were
        # lent tools. Otherwise its passive subject is the transferred item.
        item_tokens = list(objects)
        if verb == "borrow" or not objects:
            item_tokens.extend(child for child in subjects if child.dep_ in {"nsubjpass", "nsubj:pass"})
        items = {
            name for item in item_tokens for nominal in _nominals(item) if (name := _identity(nominal)) is not None
        }
        negated = any(child.dep_ == "neg" for child in children)
        agent_role = "borrower" if verb == "borrow" else "lender"

        add = partial(_add_roles, roles, negated, items)

        for subject in subjects:
            if subject.dep_ in {"nsubjpass", "nsubj:pass"}:
                if verb != "borrow" and objects:
                    add(subject, "borrower")
            else:
                add(subject, agent_role)
        # The parser can attach from/to to the direct object rather than the verb.
        arguments = [*children, *(prep for obj in objects for prep in obj.children if prep.dep_ == "prep")]
        for child in arguments:
            if child.dep_ in {"dative", "iobj"} and child.pos_ in _NOMINAL and verb != "borrow":
                add(child, "borrower")
            if child.dep_ not in {"prep", "agent", "dative"}:
                continue
            prep = child.lemma_.casefold()
            if prep == "by":
                role = agent_role
            elif prep == "from" and verb == "borrow":
                role = "lender"
            elif prep == "to" and verb != "borrow":
                role = "borrower"
            else:
                continue
            for obj in child.children:
                if obj.dep_ == "pobj":
                    add(obj, role)
    return dict(roles)


def verify_text_rewrite_loan_roles(source: str, candidate: str, *, parser: Any) -> bool:
    """Reject selected parsed English borrower/lender reversals.

    ``parser`` is a caller-owned English spaCy-compatible pipeline with complete
    dependency, POS and lemma annotations. No model is imported or downloaded.
    Use ``functools.partial`` to pass this as ``rewrite_validator``. Parser errors
    propagate, so an unavailable parse cannot become acceptance.

    Active/passive borrow, lend and loan clauses map to borrower/lender roles.
    Exact participant lemmas and available item lemmas bind the comparison.
    Unresolved pronouns and unmatched entities/events are not inferred. Matching
    roles do not prove preservation of objects, quantities, timing, negation,
    coreference, source language or general meaning. Combine with the native
    guards; this optional check is not enabled by default.
    """
    if not source.strip() or not candidate.strip():
        raise ValueError("Loan-role checks require nonempty source and candidate prose")
    if parser.lang != "en":
        raise ValueError("Loan-role checks require a caller-selected English parser")
    docs = list(parser.pipe((source, candidate)))
    if len(docs) != 2:
        raise ValueError("Loan-role parser must return one complete document per input")
    if any(doc.text != original for doc, original in zip(docs, (source, candidate), strict=True)):
        raise ValueError("Loan-role parser must preserve each exact input text")
    before, after = map(_loan_roles, docs)
    return not any(before[key].isdisjoint(after[key]) for key in before.keys() & after.keys())
