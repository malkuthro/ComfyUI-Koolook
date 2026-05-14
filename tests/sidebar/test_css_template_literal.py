"""Static guard against the recurring trap that bit issue #137 three times:
a literal backtick inside a comment in the CSS-template-literal block of
``web/sidebar/constants.js`` closes the template early and breaks the
entire sidebar at load.

The CSS is delivered as a single ``s.textContent = `\\n...\\n`;`` template
literal inside ``ensureStyle()``. Any backtick between those delimiters is
either:

  * Intentional (none exists today — the CSS surface doesn't need them).
  * Accidental (a comment that wrote ``code-style`` with double-backticks,
    or an editor's smart-quotes pass that converted them).

Either way, the file silently loads as garbage and ``ensureStyle`` ends up
bound to a CSS string instead of a function. The test below pins this:
exactly two backticks in the CSS block (open + close) and nothing in
between.

If a future change genuinely needs a backtick in CSS — e.g. a future
content selector that references a character — replace this assertion
with a more permissive one (count expected backticks explicitly). Do not
remove the test outright.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTANTS_JS = REPO_ROOT / "web" / "sidebar" / "constants.js"

# Sentinel comment lines in ``constants.js`` that frame the template
# literal. Their presence is part of the file's contract — finding them
# anchors the assertion against an obvious refactor (someone renaming
# ``ensureStyle`` or moving the CSS to its own file) rather than against
# a brittle line number.
START_MARKER = "s.textContent = `"
END_MARKER = "`;"


def test_constants_css_template_has_no_stray_backticks() -> None:
    text = CONSTANTS_JS.read_text(encoding="utf-8")

    start = text.find(START_MARKER)
    assert start != -1, (
        f"Expected ``{START_MARKER}`` in {CONSTANTS_JS.relative_to(REPO_ROOT)} — "
        "the CSS template literal's opening delimiter has moved or been removed."
    )

    # The closing delimiter is the next backtick after the opening one.
    css_block_start = start + len(START_MARKER) - 1  # index OF the opening backtick
    next_backtick = text.find("`", css_block_start + 1)
    assert next_backtick != -1, (
        f"Expected a closing backtick after the CSS template literal in "
        f"{CONSTANTS_JS.relative_to(REPO_ROOT)}; file may be truncated."
    )

    body = text[css_block_start + 1 : next_backtick]
    stray = body.count("`")
    assert stray == 0, (
        f"Found {stray} stray backtick(s) inside the CSS template literal in "
        f"{CONSTANTS_JS.relative_to(REPO_ROOT)}. A backtick inside the CSS "
        "string closes the template literal early and breaks the entire "
        "sidebar at load. Use plain quotes in comments, never backticks. "
        "See docs/maintainers/visual-harness.md for the long-form reminder."
    )
