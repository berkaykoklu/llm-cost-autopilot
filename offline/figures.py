"""Draw the two README figures from data/curve.json as standalone SVG (dark "screen" style)."""

import json
from pathlib import Path
from typing import Any

OUT = Path("docs")
BG, INK, MUTED, GRID, BOX = "#0c1224", "#e8ecf8", "#8b95b5", "#222b47", "#131a31"
CHEAP, EXPENSIVE, JEV, TFIDF = "#34c48f", "#f0a040", "#b186ff", "#8d97b0"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
STYLE = ""  # static on purpose: GitHub previews and thumbnails don't run SVG animation


def _open(w: int, h: int, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{label}" font-family="{FONT}">{STYLE}'
        f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>'
    )


def _points(series: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [(p["cost"], p["accuracy"]) for p in series]


def _at(table: list[dict[str, float]], t: float) -> tuple[float, float]:
    row = next(r for r in table if abs(r["threshold"] - t) < 1e-9)
    return row["cost"], row["accuracy"]


def picks(curve: dict[str, Any]) -> dict[str, tuple[float, float]]:
    router = _points(curve["router"])
    return {
        "all_sol": router[0],
        "all_luna": router[-1],
        "consistency": _at(curve["cascade_by_threshold"], 1.0),
        "jev": _at(curve["jev_by_threshold"], curve["jev_threshold_from_train"]),
    }


def curve_svg(curve: dict[str, Any]) -> str:
    W, H, L, R, T, B = 720, 440, 64, 28, 64, 56
    max_cost = max(float(p["cost"]) for p in curve["router"]) * 1.06

    def x(c: float) -> float:
        return L + c / max_cost * (W - L - R)

    def y(a: float) -> float:
        return T + (1 - (a - 0.54) / (0.72 - 0.54)) * (H - T - B)

    def path(pts: list[tuple[float, float]]) -> str:
        return "".join(f"{'L' if i else 'M'}{x(c):.1f},{y(a):.1f}" for i, (c, a) in enumerate(pts))

    p = picks(curve)
    s = [_open(W, H, "Cost versus accuracy on 300 held-out questions")]
    s.append(
        f'<text x="{L}" y="30" font-size="17" font-weight="700" fill="{INK}">Cost vs accuracy, 300 held-out SuperGPQA questions</text>'
    )
    s.append(
        f'<text x="{L}" y="48" font-size="12" fill="{MUTED}">every point is one threshold; thresholds for the dots were fixed before looking at these questions</text>'
    )
    for a in (0.56, 0.60, 0.64, 0.68, 0.72):
        s.append(
            f'<line x1="{L}" x2="{W - R}" y1="{y(a):.1f}" y2="{y(a):.1f}" stroke="{GRID}"/>'
            f'<text x="{L - 10}" y="{y(a) + 4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{a:.0%}</text>'
        )
    for c in (0.0, 0.2, 0.4, 0.6, 0.8):
        if c <= max_cost:
            s.append(
                f'<text x="{x(c):.1f}" y="{H - B + 20}" text-anchor="middle" font-size="12" fill="{MUTED}">${c:.1f}</text>'
            )
    s.append(
        f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle" font-size="12" fill="{MUTED}">total cost, standard API prices</text>'
    )
    s.append(
        f'<path d="{path([p["all_luna"], p["all_sol"]])}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 5"/>'
    )
    for key, color, width in (("router", TFIDF, 2), ("cascade", CHEAP, 3), ("jev", JEV, 3)):
        s.append(
            f'<path d="{path(_points(curve[key]))}" fill="none" stroke="{color}" stroke-width="{width}"/>'
        )

    def dot(pt: tuple[float, float], color: str, label: str, dx: int, dy: int) -> str:
        return (
            f'<circle cx="{x(pt[0]):.1f}" cy="{y(pt[1]):.1f}" r="5.5" fill="{color}" stroke="{BG}" stroke-width="2"/>'
            f'<text x="{x(pt[0]) + dx:.1f}" y="{y(pt[1]) + dy:.1f}" font-size="12.5" font-weight="600" fill="{INK}" '
            f'stroke="{BG}" stroke-width="5" paint-order="stroke">{label}</text>'
        )

    s.append(dot(p["all_luna"], MUTED, f"all Luna {p['all_luna'][1]:.1%}", 10, 18))
    s.append(dot(p["all_sol"], MUTED, f"all Sol {p['all_sol'][1]:.1%}", -96, -12))
    s.append(dot(p["consistency"], CHEAP, f"consistency {p['consistency'][1]:.1%}", -150, 22))
    s.append(dot(p["jev"], JEV, f"Jev {p['jev'][1]:.1%}", -30, -14))
    legend = [
        (CHEAP, "cascade: self-consistency"),
        (JEV, "cascade: Jev verifier"),
        (TFIDF, "TF-IDF router (prompt only)"),
        (MUTED, "random routing"),
    ]
    for i, (color, text) in enumerate(legend):
        yy = H - B - 84 + i * 18
        dash = ' stroke-dasharray="5 5"' if i == 3 else ""
        s.append(
            f'<line x1="{W - R - 214}" x2="{W - R - 190}" y1="{yy - 4}" y2="{yy - 4}" stroke="{color}" stroke-width="3"{dash}/>'
            f'<text x="{W - R - 182}" y="{yy}" font-size="12" fill="{MUTED}">{text}</text>'
        )
    return "".join(s) + "</svg>"


def architecture_svg(curve: dict[str, Any]) -> str:
    p = picks(curve)
    sol = p["all_sol"][0]

    def saving(key: str) -> str:
        return f"{p[key][1]:.1%} at {1 - p[key][0] / sol:.0%} lower cost"

    def box(x: int, y: int, w: int, title: str, sub: str, stroke: str = GRID) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="50" rx="10" fill="{BOX}" stroke="{stroke}" stroke-width="1.6"/>'
            f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" font-size="14" font-weight="600" fill="{INK}">{title}</text>'
            f'<text x="{x + w / 2}" y="{y + 39}" text-anchor="middle" font-size="11.5" fill="{MUTED}">{sub}</text>'
        )

    def marker(mid: str, color: str) -> str:
        return (
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
        )

    W, H = 780, 330
    s = [
        _open(W, H, "How a request flows through the proxy"),
        f"<defs>{marker('a', MUTED)}{marker('c', CHEAP)}{marker('e', EXPENSIVE)}</defs>",
    ]
    s.append(
        f'<text x="24" y="32" font-size="17" font-weight="700" fill="{INK}">One request, model: "auto"</text>'
    )
    row = 150
    for i, (title, sub) in enumerate(
        [
            ("Your app", "OpenAI SDK"),
            ("Cache", "repeat → $0"),
            ("Luna", "cheap, no reasoning"),
            ("Verifier", "5 samples agree? / Jev"),
        ]
    ):
        s.append(box(24 + i * 146, row, 124, title, sub))
        if i:
            s.append(
                f'<path d="M{24 + i * 146 - 22},{row + 25} L{24 + i * 146 - 2},{row + 25}" stroke="{MUTED}" '
                f'stroke-width="1.8" marker-end="url(#a)"/>'
            )
    s.append(box(620, 56, 140, "Serve Luna", "verifier is confident", stroke=CHEAP))
    s.append(box(620, 244, 140, "Ask Sol", "reasoning, ~20× price", stroke=EXPENSIVE))
    s.append(
        f'<path d="M586,{row + 14} C640,{row - 10} 690,140 690,110" fill="none" stroke="{CHEAP}" stroke-width="2.6" marker-end="url(#c)"/>'
    )
    s.append(
        f'<path d="M586,{row + 36} C640,{row + 60} 690,210 690,240" fill="none" stroke="{EXPENSIVE}" stroke-width="2.6" marker-end="url(#e)"/>'
    )
    s.append(
        f'<text x="24" y="66" font-size="12.5" fill="{MUTED}"><tspan fill="{CHEAP}">■</tspan> consistency: {saving("consistency")}</text>'
    )
    s.append(
        f'<text x="24" y="86" font-size="12.5" fill="{MUTED}"><tspan fill="{JEV}">■</tspan> Jev verifier: {saving("jev")}</text>'
    )
    s.append(
        f'<text x="24" y="106" font-size="12.5" fill="{MUTED}">all Sol: {p["all_sol"][1]:.1%}</text>'
    )
    s.append(
        f'<text x="24" y="{H - 24}" font-size="11.5" fill="{MUTED}">every request is logged with its cost and what all-Sol would have cost</text>'
    )
    return "".join(s) + "</svg>"


def main() -> None:
    curve = json.loads(Path("data/curve.json").read_text())
    OUT.mkdir(exist_ok=True)
    (OUT / "curve.svg").write_text(curve_svg(curve))
    (OUT / "architecture.svg").write_text(architecture_svg(curve))
    print(f"wrote {OUT / 'curve.svg'} and {OUT / 'architecture.svg'}")


if __name__ == "__main__":
    main()
