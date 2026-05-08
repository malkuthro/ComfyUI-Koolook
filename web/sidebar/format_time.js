// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Local-timezone timestamp formatter — combined relative + 24-hour absolute.
//
// Output cases (in priority order):
//   • <1 min ago      → "just now"
//   • <60 min ago     → "12 min ago"
//   • same calendar day (local) → "today 14:32"
//   • previous day            → "yesterday 23:58"
//   • else, current year      → "May 6 14:32"
//   • else                    → "May 6 2025 14:32"
//
// Why 24-hour: the previous `en-US` 12-hour format produced "12:56 AM" for
// timestamps just past midnight, which reads visually like noon to most
// people (the US "12 AM = midnight" convention is a known UX trap).
// Manually building HH:MM from `getHours()` / `getMinutes()` sidesteps the
// `hour12: false` quirk in some locales (e.g. en-US emitting "24:00").
//
// Why "today" / "yesterday" tier: makes the most-relevant cases (recent
// snapshots, recent autosaves) read at a glance without making the eye
// parse a full date.
// =============================================================================

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function _hhmm(d) {
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
}

export function formatLocalStamp(d, now = new Date()) {
    if (!(d instanceof Date) || isNaN(d.getTime())) return "";
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    // Future timestamps (clock skew, browser/server drift): fall through to
    // absolute formatting rather than emitting "-3 min ago".
    if (diffMin >= 0 && diffMin < 1) return "just now";
    if (diffMin >= 1 && diffMin < 60) return diffMin + " min ago";

    const time = _hhmm(d);
    const startOfToday = new Date(now);
    startOfToday.setHours(0, 0, 0, 0);
    const startOfYesterday = new Date(startOfToday);
    startOfYesterday.setDate(startOfYesterday.getDate() - 1);

    if (d >= startOfToday) return "today " + time;
    if (d >= startOfYesterday) return "yesterday " + time;

    const datePart = MONTHS[d.getMonth()] + " " + d.getDate();
    const yearPart = d.getFullYear() !== now.getFullYear() ? " " + d.getFullYear() : "";
    return datePart + yearPart + " " + time;
}
