"""Map measured numerical reproducibility onto the SPEC verdict enums."""

from __future__ import annotations


def classify_determinism_result(
    *,
    bit_exact_rate: float,
    task_stable: bool | None,
    effective_level: str,
    requested_level: str,
    error: str | None = None,
) -> str:
    """Headline reproducibility verdict (SPEC §Determinism Result).

    `task_stable` is the task-level signal (identical greedy tokens for LLMs,
    identical argmax for vision); None when not applicable.
    """
    if error is not None:
        return "ERROR"
    # Requested strict-style determinism the backend cannot enforce.
    if requested_level == "FULL_DET" and effective_level != "FULL_DET":
        return "UNSUPPORTED"
    if bit_exact_rate >= 100.0:
        return "BIT_EXACT"
    if task_stable:
        return "OUTPUT_STABLE"
    return "DRIFTED"


def determinism_achieved(*, effective_level: str, bit_exact_rate: float, error: str | None) -> str:
    """Safety column for the `strict` run only (SPEC §Determinism Achieved).

    YES = bit-identical every run on this hardware; NO = still varied;
    UNSUPPORTED = strict controls cannot be enforced (e.g. MLX) so it can't be asserted.
    Carries no cost information by construction.
    """
    if effective_level != "FULL_DET" or error is not None:
        return "UNSUPPORTED"
    return "YES" if bit_exact_rate >= 100.0 else "NO"
