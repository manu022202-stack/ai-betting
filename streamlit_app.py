import os
import math
import requests
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================

SPORT = "baseball_mlb"
REGION = "us"
ODDS_FORMAT = "american"
BASE_URL = "https://api.the-odds-api.com/v4/sports"

GAME_MARKETS = ["h2h", "spreads", "totals"]

st.set_page_config(
    page_title="MLB Prop Probability Lab",
    page_icon="⚾",
    layout="wide"
)

# =========================
# STYLING
# =========================

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    .stMarkdown, .stText, p, label {
        color: #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%);
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
        margin-bottom: .75rem;
    }
    .small-muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .good {
        color: #22c55e;
        font-weight: 700;
    }
    .warn {
        color: #f59e0b;
        font-weight: 700;
    }
    .bad {
        color: #ef4444;
        font-weight: 700;
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# UTILITY FUNCTIONS
# =========================

def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)

def decimal_to_american(decimal_odds):
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))

def probability_to_fair_american(prob):
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return int(round(-(prob / (1 - prob)) * 100))
    return int(round(((1 - prob) / prob) * 100))

def american_to_implied(odds):
    return 1 / american_to_decimal(odds)

def clamp(x, low, high):
    return max(low, min(high, x))

def poisson_over_probability(mean, line):
    """
    P(X > line) for a Poisson count.
    Works well enough as a starter for strikeouts, hits, total bases style counts.
    For 0.5 line, over means >=1.
    For 1.5 line, over means >=2.
    """
    threshold = math.floor(line) + 1
    cumulative = 0.0
    for k in range(threshold):
        cumulative += math.exp(-mean) * (mean ** k) / math.factorial(k)
    return clamp(1 - cumulative, 0.01, 0.99)

def confidence_label(prob, prop_type):
    if prop_type == "HR":
        if prob >= 0.16:
            return "Strong HR longshot"
        if prob >= 0.11:
            return "Playable HR lean"
        return "Low-prob HR"
    if prob >= 0.68:
        return "Strong projection"
    if prob >= 0.58:
        return "Good projection"
    if prob >= 0.52:
        return "Lean"
    return "Avoid / thin"

def sportsbook_value_note(prob, fair_odds):
    if fair_odds < 0:
        return f"Only consider if sportsbook price is better than {fair_odds}, e.g. less juiced like -150 instead of -190."
    return f"Only consider if sportsbook price is better than +{fair_odds}, e.g. +150 instead of +120."

# =========================
# ODDS API
# =========================

def fetch_game_odds(api_key):
    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": ",".join(GAME_MARKETS),
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_game_odds(raw):
    rows = []
    for game in raw:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_name = f"{away} @ {home}"
        for book in game.get("bookmakers", []):
            book_name = book.get("title", "")
            for market in book.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    odds = outcome.get("price")
                    if odds is None:
                        continue
                    point = outcome.get("point")
                    label = outcome.get("name", "")
                    if market_key == "h2h":
                        market_name = "Moneyline"
                        pick = label
                    elif market_key == "spreads":
                        market_name = "Run line"
                        pick = f"{label} {point:+g}" if point is not None else label
                    elif market_key == "totals":
                        market_name = "Total"
                        pick = f"{label} {point:g}" if point is not None else label
                    else:
                        market_name = market_key
                        pick = label
                    rows.append({
                        "game": game_name,
                        "sportsbook": book_name,
                        "market": market_name,
                        "pick": pick,
                        "odds": int(odds),
                        "line": point,
                        "implied_prob": american_to_implied(odds),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["decimal"] = df["odds"].apply(american_to_decimal)
    return df

# =========================
# PROP MODELS
# =========================

def hitter_projection(
    prop,
    line,
    batting_order,
    handedness_edge,
    recent_form,
    pitcher_quality,
    pitcher_contact_allowed,
    park_factor,
    weather_factor,
    team_total,
):
    """
    Returns projected probability and model notes.

    This is a starter projection model without prop odds.
    It does not fetch individual Statcast player data yet.
    The sliders are a way to encode the same inputs:
    - matchup
    - recent form
    - pitcher quality/contact
    - stadium/weather
    - team context
    """

    notes = []

    # Baselines are intentionally conservative.
    if prop == "Hit Over":
        mean = 0.92
        mean += (5 - batting_order) * 0.045
        mean += handedness_edge * 0.035
        mean += recent_form * 0.040
        mean += pitcher_contact_allowed * 0.045
        mean += park_factor * 0.030
        mean += weather_factor * 0.020
        mean += (team_total - 4.3) * 0.055
        mean -= pitcher_quality * 0.030
        prob = poisson_over_probability(max(0.20, mean), line)

    elif prop == "Total Bases Over":
        mean = 1.38
        mean += (5 - batting_order) * 0.060
        mean += handedness_edge * 0.060
        mean += recent_form * 0.070
        mean += pitcher_contact_allowed * 0.075
        mean += park_factor * 0.055
        mean += weather_factor * 0.045
        mean += (team_total - 4.3) * 0.080
        mean -= pitcher_quality * 0.045
        prob = poisson_over_probability(max(0.15, mean), line)

    elif prop == "RBI Over":
        mean = 0.42
        mean += max(0, 5 - batting_order) * 0.020
        mean += handedness_edge * 0.025
        mean += recent_form * 0.025
        mean += pitcher_contact_allowed * 0.035
        mean += park_factor * 0.020
        mean += weather_factor * 0.015
        mean += (team_total - 4.3) * 0.075
        mean -= pitcher_quality * 0.020
        prob = poisson_over_probability(max(0.08, mean), line)

    elif prop == "Run Over":
        mean = 0.50
        mean += max(0, 6 - batting_order) * 0.030
        mean += handedness_edge * 0.020
        mean += recent_form * 0.025
        mean += pitcher_contact_allowed * 0.025
        mean += park_factor * 0.020
        mean += weather_factor * 0.020
        mean += (team_total - 4.3) * 0.080
        mean -= pitcher_quality * 0.015
        prob = poisson_over_probability(max(0.08, mean), line)

    elif prop == "Home Run":
        # HR probability is modeled directly, not Poisson over.
        prob = 0.055
        prob += max(0, 5 - batting_order) * 0.004
        prob += handedness_edge * 0.012
        prob += recent_form * 0.008
        prob += pitcher_contact_allowed * 0.014
        prob += park_factor * 0.010
        prob += weather_factor * 0.010
        prob += (team_total - 4.3) * 0.010
        prob -= pitcher_quality * 0.007
        prob = clamp(prob, 0.005, 0.32)

    else:
        prob = 0.50

    if handedness_edge > 0:
        notes.append("handedness edge")
    if recent_form > 0:
        notes.append("positive recent form")
    if pitcher_contact_allowed > 0:
        notes.append("pitcher allows contact/damage")
    if park_factor > 0:
        notes.append("positive park factor")
    if weather_factor > 0:
        notes.append("weather boost")
    if pitcher_quality > 0:
        notes.append("pitcher quality suppresses projection")

    return prob, ", ".join(notes) if notes else "neutral setup"

def pitcher_projection(
    prop,
    line,
    pitcher_k_skill,
    opponent_k_rate,
    pitch_count,
    pitcher_form,
    umpire_k_boost,
    opponent_power,
    park_factor,
    weather_factor,
):
    notes = []

    if prop == "Strikeouts Over":
        mean = 4.9
        mean += pitcher_k_skill * 0.38
        mean += opponent_k_rate * 0.32
        mean += (pitch_count - 90) * 0.035
        mean += pitcher_form * 0.22
        mean += umpire_k_boost * 0.16
        mean -= opponent_power * 0.10
        prob = poisson_over_probability(max(1.0, mean), line)

    elif prop == "Outs Recorded Over":
        mean = 16.2
        mean += (pitch_count - 90) * 0.090
        mean += pitcher_form * 0.42
        mean -= opponent_power * 0.35
        mean -= park_factor * 0.20
        mean -= weather_factor * 0.12
        mean += pitcher_k_skill * 0.12
        prob = poisson_over_probability(max(6.0, mean), line)

    elif prop == "Earned Runs Under":
        # Probability of staying UNDER the line.
        mean = 2.55
        mean -= pitcher_k_skill * 0.16
        mean -= pitcher_form * 0.18
        mean += opponent_power * 0.28
        mean += park_factor * 0.14
        mean += weather_factor * 0.10

        # P(X < line). For under 2.5, threshold means X <= 2.
        upper = math.ceil(line) - 1
        cumulative = 0.0
        for k in range(max(0, upper) + 1):
            cumulative += math.exp(-mean) * (mean ** k) / math.factorial(k)
        prob = clamp(cumulative, 0.01, 0.99)

    else:
        prob = 0.50

    if pitcher_k_skill > 0:
        notes.append("pitcher K skill boost")
    if opponent_k_rate > 0:
        notes.append("opponent strikeout tendency")
    if pitcher_form > 0:
        notes.append("positive recent pitcher form")
    if umpire_k_boost > 0:
        notes.append("umpire boost")
    if opponent_power > 0:
        notes.append("dangerous opposing offense")
    if park_factor > 0:
        notes.append("hitter-friendly park")
    if weather_factor > 0:
        notes.append("weather helps offense")

    return prob, ", ".join(notes) if notes else "neutral setup"

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "The Odds API key",
        type="password",
        value=os.getenv("ODDS_API_KEY", ""),
        help="Only needed for game odds/team total context. Props do not require prop odds."
    )

    st.divider()

    mode = st.radio(
        "Tool mode",
        ["Player prop probability", "Game odds scanner"],
        index=0
    )

    st.divider()

    st.caption("Quick guide")
    st.markdown(
        """
        **Probability** = how likely the prop is.  
        **Fair odds** = price you would need.  
        If sportsbook odds are worse than fair odds, skip it.
        """
    )

# =========================
# HEADER
# =========================

st.markdown(
    """
    <div class="metric-card">
        <h1>⚾ MLB Prop Probability Lab</h1>
        <p class="small-muted">
        Project player prop probabilities without needing prop odds. Use the fair odds as your comparison number when checking FanDuel, DraftKings, BetMGM, etc.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "No bet is safe or guaranteed. This app estimates probability and fair price. "
    "It cannot confirm profitability unless you compare the fair odds to the sportsbook's actual odds."
)

# =========================
# PLAYER PROP MODE
# =========================

if mode == "Player prop probability":
    tab1, tab2, tab3 = st.tabs(["🥎 Hitter props", "🔥 Pitcher props", "📘 How to use fair odds"])

    with tab1:
        st.subheader("Hitter prop calculator")

        colA, colB = st.columns([1.1, 1])

        with colA:
            player_name = st.text_input("Player name", placeholder="Example: Juan Soto")
            prop = st.selectbox(
                "Prop type",
                ["Hit Over", "Total Bases Over", "RBI Over", "Run Over", "Home Run"]
            )

            default_line = 0.5 if prop in ["Hit Over", "RBI Over", "Run Over"] else 1.5
            if prop == "Home Run":
                default_line = 0.5

            line = st.number_input("Prop line", value=float(default_line), step=0.5)

            batting_order = st.slider("Projected batting order spot", 1, 9, 3)

            team_total = st.slider(
                "Estimated team total",
                2.0, 7.5, 4.3, 0.1,
                help="Use game total / implied team total if available. Higher team totals boost runs/RBI/overall hitting props."
            )

        with colB:
            st.markdown("#### Matchup inputs")
            handedness_edge = st.slider("Batter handedness/split edge", -3, 3, 0)
            recent_form = st.slider("Recent form / rolling contact", -3, 3, 0)
            pitcher_quality = st.slider("Opposing pitcher quality", -3, 3, 0)
            pitcher_contact_allowed = st.slider("Pitcher allows contact/damage", -3, 3, 0)
            park_factor = st.slider("Stadium factor for this prop", -3, 3, 0)
            weather_factor = st.slider("Weather/wind factor", -3, 3, 0)

        prob, notes = hitter_projection(
            prop=prop,
            line=line,
            batting_order=batting_order,
            handedness_edge=handedness_edge,
            recent_form=recent_form,
            pitcher_quality=pitcher_quality,
            pitcher_contact_allowed=pitcher_contact_allowed,
            park_factor=park_factor,
            weather_factor=weather_factor,
            team_total=team_total,
        )

        fair_odds = probability_to_fair_american(prob)
        label = confidence_label(prob, "HR" if prop == "Home Run" else "Hitter")

        c1, c2, c3 = st.columns(3)
        c1.metric("Projected probability", f"{prob*100:.1f}%")
        c2.metric("Fair odds", f"{fair_odds:+d}")
        c3.metric("Confidence", label)

        st.markdown(
            f"""
            <div class="metric-card">
            <b>Read:</b> {player_name or "This player"} {prop} {line:g}<br>
            <span class="small-muted">Model notes: {notes}</span><br><br>
            <span class="good">{sportsbook_value_note(prob, fair_odds)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        result = pd.DataFrame([{
            "player": player_name,
            "prop": prop,
            "line": line,
            "projected_probability": round(prob, 4),
            "fair_odds": fair_odds,
            "confidence": label,
            "notes": notes,
        }])

        st.download_button(
            "Download hitter projection CSV",
            result.to_csv(index=False).encode("utf-8"),
            "hitter_prop_projection.csv",
            "text/csv"
        )

    with tab2:
        st.subheader("Pitcher prop calculator")

        colA, colB = st.columns([1.1, 1])

        with colA:
            pitcher_name = st.text_input("Pitcher name", placeholder="Example: Zack Wheeler")
            pitcher_prop = st.selectbox(
                "Pitcher prop type",
                ["Strikeouts Over", "Outs Recorded Over", "Earned Runs Under"]
            )

            default_pitcher_line = 5.5 if pitcher_prop == "Strikeouts Over" else 17.5
            if pitcher_prop == "Earned Runs Under":
                default_pitcher_line = 2.5

            pitcher_line = st.number_input("Prop line", value=float(default_pitcher_line), step=0.5)
            pitch_count = st.slider("Projected pitch count", 60, 115, 90)

        with colB:
            st.markdown("#### Pitching matchup inputs")
            pitcher_k_skill = st.slider("Pitcher K skill", -3, 3, 0)
            opponent_k_rate = st.slider("Opponent strikeout tendency", -3, 3, 0)
            pitcher_form = st.slider("Recent pitcher form", -3, 3, 0)
            umpire_k_boost = st.slider("Umpire strike zone/K boost", -3, 3, 0)
            opponent_power = st.slider("Opponent offensive danger", -3, 3, 0)
            p_park_factor = st.slider("Park favors offense", -3, 3, 0)
            p_weather_factor = st.slider("Weather favors offense", -3, 3, 0)

        p_prob, p_notes = pitcher_projection(
            prop=pitcher_prop,
            line=pitcher_line,
            pitcher_k_skill=pitcher_k_skill,
            opponent_k_rate=opponent_k_rate,
            pitch_count=pitch_count,
            pitcher_form=pitcher_form,
            umpire_k_boost=umpire_k_boost,
            opponent_power=opponent_power,
            park_factor=p_park_factor,
            weather_factor=p_weather_factor,
        )

        p_fair_odds = probability_to_fair_american(p_prob)
        p_label = confidence_label(p_prob, "Pitcher")

        c1, c2, c3 = st.columns(3)
        c1.metric("Projected probability", f"{p_prob*100:.1f}%")
        c2.metric("Fair odds", f"{p_fair_odds:+d}")
        c3.metric("Confidence", p_label)

        st.markdown(
            f"""
            <div class="metric-card">
            <b>Read:</b> {pitcher_name or "This pitcher"} {pitcher_prop} {pitcher_line:g}<br>
            <span class="small-muted">Model notes: {p_notes}</span><br><br>
            <span class="good">{sportsbook_value_note(p_prob, p_fair_odds)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        pitcher_result = pd.DataFrame([{
            "pitcher": pitcher_name,
            "prop": pitcher_prop,
            "line": pitcher_line,
            "projected_probability": round(p_prob, 4),
            "fair_odds": p_fair_odds,
            "confidence": p_label,
            "notes": p_notes,
        }])

        st.download_button(
            "Download pitcher projection CSV",
            pitcher_result.to_csv(index=False).encode("utf-8"),
            "pitcher_prop_projection.csv",
            "text/csv"
        )

    with tab3:
        st.subheader("How to use fair odds")

        st.markdown(
            """
            The app gives you a **fair odds** number. That is the break-even price based on the model probability.

            Examples:

            | Projection | Fair odds | What you want |
            |---:|---:|---|
            | 60% | -150 | Better than -150, such as -130 or +100 |
            | 50% | +100 | Better than +100 |
            | 40% | +150 | Better than +150, such as +170 |
            | 25% | +300 | Better than +300, such as +350 |

            For props, the best workflow is:

            1. Use the app to project probability.
            2. Check your sportsbook for the actual prop line and price.
            3. Only consider it if the sportsbook price is **better than fair odds**.
            4. Avoid parlays unless each individual leg has value.
            """
        )

# =========================
# GAME ODDS SCANNER MODE
# =========================

else:
    st.subheader("Game odds scanner")

    if not api_key:
        st.info("Enter your Odds API key in the sidebar to scan moneylines, run lines, and totals.")
        st.stop()

    with st.spinner("Pulling game odds..."):
        try:
            raw = fetch_game_odds(api_key)
            df = parse_game_odds(raw)
        except Exception as e:
            st.error(f"Could not pull odds: {e}")
            st.stop()

    if df.empty:
        st.info("No game odds found right now.")
        st.stop()

    books = ["Best available"] + sorted(df["sportsbook"].dropna().unique().tolist())
    selected_book = st.selectbox("Sportsbook", books)

    if selected_book == "Best available":
        view = df.sort_values("decimal", ascending=False).drop_duplicates(["game", "market", "pick"])
    else:
        view = df[df["sportsbook"] == selected_book].copy()

    view["implied_prob"] = (view["implied_prob"] * 100).round(1).astype(str) + "%"

    st.dataframe(
        view[["game", "sportsbook", "market", "pick", "odds", "line", "implied_prob"]].head(200),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download game odds CSV",
        view.to_csv(index=False).encode("utf-8"),
        "game_odds.csv",
        "text/csv"
    )
