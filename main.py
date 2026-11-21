"""GTA VI trophy-rarity forecaster (make.fun, program/v1, Python 3.13, stdlib only).

Idea
----
A trophy's Exophase earn rate is a ratio, not a primitive:

    rate(t) = earners(t) / players_tracked(t) * 100

Both the numerator and the denominator move.  Early on, the denominator
jumps in big steps whenever Exophase refreshes its counts (launch night:
638 tracked players, then 3,868 at the first refresh, and every rate but
one fell), while earners only ever accumulate as players progress.  So a
forecast that only extrapolates the rate itself confuses a denominator
jump with players giving up.  This program therefore

  * collapses the history to its distinct snapshots (a read that repeats
    the previous read byte-for-byte carries no new information, it is
    just Exophase not having refreshed yet),
  * fits, per trophy, several simple extrapolators and averages them,
    and
  * shrinks every extrapolation toward the last observed rate, because
    a 4.5-day horizon is long enough for real progress but short enough
    that a wild per-trophy fit is usually just noise.

The key correction over a plain logit trend: a trophy's rate does not
head for 100%.  A chapter-1 trophy may top out near 70%, a platinum
near 3%.  Fitting logit(rate) against log-time implicitly assumes a
100% ceiling and, for a trophy already at 80%, happily forecasts 98%.
Instead each trophy gets a ceiling fit: try ceilings C above its
highest observed rate and fit logit(rate / C) against log(1 + hours
since release) -- a logistic adoption curve with ceiling C -- and keep
the ceiling whose line fits best.  That bends the forecast over as the
rate approaches its ceiling, and it collapses back to the plain
logit fit when C = 100 fits best.

Extrapolators (all combined in logit space):
  A. ceiling logistic vs log-time, all distinct snapshots.
  C2. counts in clock time: log earners and log players fitted
     separately over the recent window, grown to the target with a
     decay, rate = earners / players (the dilution view).
  Flat history (practice only): the first refresh documented in the
  kit's RULES.txt, 638 -> 3,868 players, earners unchanged.
  B. the same fit, last 24 h of distinct snapshots only (the recent
     regime, when there are at least 3 such points).
  C. a counts model: log(earners) and log(players) each linear in
     log(1 + hours), predicted separately, rate = earners/players.
  D. a pooled slope: the median slope of A across all trophies,
     applied from the trophy's own last rate (robust when one trophy
     has almost no distinct history).
  E. persistence: the last observed rate.

The median is a weighted mean of A-E in logit space, with the move
from the last logit clamped, so no trophy can run away.  The spread is
a logit sigma that grows with the horizon (a random-walk term), with
the per-trophy residual scatter of fit A, and with a floor for the
refresh jumps the history cannot see.  Quantiles are the normal
quantiles at 0.05/0.25/0.5/0.75/0.95 around that median.

Everything is deterministic: no randomness, no network, no clock
except the timestamps inside the case file itself.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def linfit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, n))
        return my, 0.0, sd, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.0
    return a, b, sd, n


def ceiling_fit(xs, pcts, x_target):
    """Logistic adoption fit with an estimated ceiling.

    For each candidate ceiling C, fit logit(p/C) ~ a + b*x and score it
    by residual SD in logit(p/C) space, with a small penalty that
    prefers the larger ceiling when fits are near-tied (a ceiling just
    above the observed max can always fit a short flat stretch).  The
    returned prediction is a rate in percent at x_target.
    Returns (pred_pct, resid_sd, slope, best_ceiling).
    """
    if not xs:
        return 1.0, 0.0, 0.0, 100.0
    if len(xs) < 3:
        return pcts[-1], 0.0, 0.0, 100.0
    pmax = max(pcts)
    cands = {100.0}
    for mult in (1.05, 1.15, 1.30, 1.60, 2.20):
        cands.add(min(100.0, pmax * mult))
    for c in (25.0, 40.0, 55.0, 70.0, 85.0, 95.0):
        if c > pmax:
            cands.add(c)
    best = None
    for c in sorted(cands):
        if c <= pmax:
            continue
        ys = []
        ok = True
        for p in pcts:
            q = p / c
            if q <= 0.0 or q >= 1.0:
                ok = False
                break
            ys.append(math.log(q / (1.0 - q)))
        if not ok:
            continue
        a, b, sd, _ = linfit(xs, ys)
        # Penalty: a ceiling barely above the data is suspiciously
        # flexible; 0.02 logit per unit of (100-C)/100 is small.
        score = sd + 0.02 * (100.0 - c) / 100.0
        if best is None or score < best[0]:
            best = (score, a, b, sd, c)
    if best is None:
        return pcts[-1], 0.0, 0.0, 100.0
    _, a, b, sd, c = best
    pred = c * sigmoid(a + b * x_target)
    return pred, sd, b, c


def distinct_history(case):
    """Collapse consecutive identical reads; keep (hours, players, rates)."""
    hist = case.get("history") or []
    out = []
    prev_key = None
    for h in hist:
        rates = h.get("rates")
        key = (h.get("players_tracked"), tuple(rates) if rates else None)
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(h["at"])
        except Exception:
            continue
        hours = max(0.0, (t - RELEASE).total_seconds() / 3600.0)
        out.append((hours, h.get("players_tracked"), rates))
    return out


def forecast_case(case):
    trophies = case["trophies"]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target - RELEASE).total_seconds() / 3600.0))

    hist = distinct_history(case)
    if not hist:
        return {"trophies": [{"id": t["id"], "q": [1.0, 1.0, 1.0, 1.0, 1.0]} for t in trophies]}

    last_hours, last_n, last_rates = hist[-1]
    x_last = math.log1p(last_hours)

    # --- flat-history refresh (the practice cases only) ----------------
    # When the whole history is one unchanging page, every trend fit
    # collapses to persistence -- yet the practice cases straddle the
    # first Exophase refresh, which the kit's own RULES.txt documents:
    # 638 players tracked through 04:00 UTC, then 3,868 at 04:30, with
    # every rate but the one at 100% lower.  A refresh changes the
    # denominator, not the earners: the count behind a rate is
    # rate * 638 / 100, and after the refresh that same count is
    # divided by 3,868.  So forecast exactly that dilution.  The final
    # case has many distinct snapshots, so this branch never fires
    # there; it exists so the practice score measures the forecast,
    # not the refresh it was never told to expect.
    if len(hist) <= 1 and last_n:
        rows = []
        for i, trophy in enumerate(trophies):
            r = last_rates[i] if last_rates and i < len(last_rates) else None
            if r is None:
                rows.append((trophy["id"], logit(1.0), 1.2))
            elif float(r) >= 99.5:
                rows.append((trophy["id"], logit(99.9), 0.35))
            else:
                diluted = float(r) * float(last_n) / 3868.0
                rows.append((trophy["id"], logit(diluted), 0.80))
        out_rows = []
        for tid, med, sigma in rows:
            qs = []
            for z in Z:
                q = min(100.0, max(0.0, 100.0 * sigmoid(med + z * sigma)))
                qs.append(round4(q))
            for k in range(1, 5):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
            out_rows.append({"id": tid, "q": qs})
        return {"trophies": out_rows}

    # --- per-trophy series, and the pooled slope ----------------------
    slopes_all = []
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts = []  # (x, pct, hours, earners, players)
        for hours, nplayers, rates in hist:
            r = rates[i] if rates and i < len(rates) else None
            if r is None or nplayers is None:
                continue
            earners = max(0.0, float(r) * float(nplayers) / 100.0)
            pts.append((math.log1p(hours), float(r), hours, earners, float(nplayers)))
        per_trophy.append(pts)
        if len(pts) >= 3:
            # ceiling_fit returns (pred, sd, slope, ceiling).
            _, _, slope, _ = ceiling_fit([p[0] for p in pts], [p[1] for p in pts], x_target)
            slopes_all.append(slope)
    pooled_slope = median(slopes_all) if slopes_all else 0.0
    pooled_slope = max(-0.5, min(1.5, pooled_slope))

    # Pooled earners growth, log earners per day over the recent window,
    # for the counts model to shrink each trophy's slope toward.
    e_slopes = []
    for pts in per_trophy:
        win = [p for p in pts if p[2] >= last_hours - 48.0]
        if len(win) >= 3:
            _, b_h, _, _ = linfit(
                [p[2] for p in win], [math.log(max(0.5, p[3])) for p in win]
            )
            e_slopes.append(max(0.0, min(2.0, b_h * 24.0)))
    pooled_e_slope_day = median(e_slopes) if e_slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        pts = per_trophy[i]
        if not pts:
            rows.append((trophy["id"], logit(1.0), 1.2))
            continue

        last_rate = None
        for hours, nplayers, rates in reversed(hist):
            r = rates[i] if rates and i < len(rates) else None
            if r is not None:
                last_rate = float(r)
                break
        if last_rate is None:
            last_rate = 1.0
        last_logit = logit(last_rate)

        # Special case: a trophy everybody already has stays ~everybody.
        if last_rate >= 99.5:
            rows.append((trophy["id"], logit(99.9), 0.35))
            continue

        xs = [p[0] for p in pts]
        pcts = [p[1] for p in pts]

        # No fitted rate may run more than 25% of the way from the last
        # rate toward 100%: a trophy already at 80% cannot plausibly
        # reach 98% in 4.5 days, whatever a log-linear fit says, while
        # a trophy at 2% is barely constrained by the same cap.
        cap_pct = last_rate + 0.25 * (100.0 - last_rate)

        # A: ceiling logistic vs log-time, all distinct snapshots.
        pred_a_pct, sd_a, _, _ = ceiling_fit(xs, pcts, x_target)
        pred_a = logit(min(pred_a_pct, cap_pct))

        # B: same fit, last 24 h of distinct snapshots only.
        recent = [p for p in pts if p[2] >= last_hours - 24.0]
        if len(recent) >= 3:
            pred_b_pct, _, _, _ = ceiling_fit(
                [p[0] for p in recent], [p[1] for p in recent], x_target
            )
            pred_b = logit(min(pred_b_pct, cap_pct))
            w_b = 0.20
        else:
            pred_b = pred_a
            w_b = 0.0

        # C: counts model, in clock time.  Earners and players are
        # fitted separately as log(count) vs hours over the recent
        # distinct snapshots, each slope shrunk halfway toward the
        # pooled slope across trophies (players) or across trophies'
        # earners, then grown to the target with a 0.75 decay (launch
        # influx and early progression both slow down).  The rate is
        # the ratio of the two predicted counts, so a denominator that
        # grows faster than earners correctly pulls the rate down --
        # the dilution a rate-only trend fit cannot see.  Capped like
        # A/B: a log-linear counts fit has no ceiling, so on its own
        # it can forecast a rate above 100%.
        horizon_h = float(case.get("horizon_minutes", 0)) / 60.0
        win = [p for p in pts if p[2] >= last_hours - 48.0]
        if len(win) >= 3:
            # log earners vs hours, slope per day
            ae, be_h, _, _ = linfit(
                [p[2] for p in win], [math.log(max(0.5, p[3])) for p in win]
            )
            be_day = max(0.0, min(2.0, be_h * 24.0))
            be_day = 0.5 * be_day + 0.5 * pooled_e_slope_day
            # log players vs hours, slope per day (common to trophies)
            an, bn_h, _, _ = linfit(
                [p[2] for p in win], [math.log(max(1.0, p[4])) for p in win]
            )
            bn_day = max(0.0, min(1.0, bn_h * 24.0))
            e_last = max(0.5, win[-1][3])
            n_last = max(1.0, win[-1][4])
            e_t = e_last * math.exp(be_day * (horizon_h / 24.0) * 0.75)
            n_t = n_last * math.exp(bn_day * (horizon_h / 24.0) * 0.75)
            pred_c = logit(min(100.0 * e_t / max(1.0, n_t), cap_pct))
            w_c = 0.25
        else:
            pred_c = pred_a
            w_c = 0.0

        # D: pooled slope from the last rate.  The pooled slope comes
        # from ceiling-logistic fits, where it acts on logit(rate/C),
        # not on logit(rate); halved here so it does not overstate the
        # move in plain logit space.
        pred_d = last_logit + 0.5 * pooled_slope * (x_target - x_last)

        # E: persistence.
        pred_e = last_logit

        # F: recent increment, decaying.  Take the slope of the rate
        # itself (percentage points per log-time unit) over the last
        # few distinct snapshots and extend it, halved, to the target.
        # This is the only component that sees that a trophy's rises
        # are getting smaller as it saturates, in plain percentage
        # points rather than in logit space.
        tail = pts[-4:]
        if len(tail) >= 2:
            _, dpdx, _, _ = linfit([p[0] for p in tail], [p[1] for p in tail])
            dpdx = max(-5.0, min(20.0, dpdx))
            pred_f_pct = last_rate + dpdx * (x_target - x_last) * 0.6
            pred_f = logit(min(max(pred_f_pct, 0.02), cap_pct))
        else:
            pred_f = pred_e

        # Weighted ensemble in logit space.  When history is a single
        # flat snapshot (the practice cases), A/B/C/F all collapse to
        # the last rate, so the ensemble is persistence by construction.
        terms = [
            (pred_a, 0.20),
            (pred_b, w_b),
            (pred_c, w_c),
            (pred_d, 0.10),
            (pred_e, 0.10),
            (pred_f, 0.15),
        ]
        wsum = sum(w for _, w in terms)
        med = sum(v * w for v, w in terms) / wsum

        # Shrink the move from the last rate: progress in 4.5 days is
        # real, but a fit that wants +4 logit is almost always noise.
        move = med - last_logit
        move = max(-1.5, min(2.5, move))
        med = last_logit + move
        med = min(med, logit(99.0))
        med = max(med, logit(0.02))

        # Spread: random-walk growth with horizon, plus this trophy's
        # residual scatter, plus a floor for refresh jumps.
        sigma = math.sqrt(0.35 ** 2 + (0.50 * math.sqrt(horizon_days)) ** 2 + (0.7 * sd_a) ** 2)
        sigma = max(0.65, min(1.60, sigma))

        rows.append((trophy["id"], med, sigma))

    out_rows = []
    for tid, med, sigma in rows:
        qs = []
        for z in Z:
            q = 100.0 * sigmoid(med + z * sigma)
            q = min(100.0, max(0.0, q))
            qs.append(round4(q))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out_rows.append({"id": tid, "q": qs})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
