import os
import math
import itertools
import requests
import pandas as pd
import streamlit as st

SPORT = "baseball_mlb"
REGION = "us"
ODDS_FORMAT = "american"
BASE_URL = "https://api.the-odds-api.com/v4/sports"

GAME_MARKETS = ["h2h", "spreads", "totals"]

st.set_page_config(page_title="MLB Bet Analyzer", page_icon="⚾", layout="centered")

st.title("⚾ MLB Bet Analyzer")
st.caption("Mobile-friendly scanner for MLB moneyline, run line, and totals.")

st.warning(
    "No bet is guaranteed or truly safe. This tool estimates value/risk and should be used with bankroll discipline. "
    "Verify starters, lineups, injuries, weather, and odds before betting."
)

def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)

def american_to_implied(odds):
    return 1 / american_to_decimal(odds)

def expected_value(probability, decimal_odds):
    return probability * (decimal_odds - 1) - (1 - probability)

def classify(edge, odds, projected_prob):
    if edge >= 0.045 and projected_prob >= 0.56 and -175 <= odds <= 150:
        return "Best value"
    if edge >= 0.025 and projected_prob >= 0.53 and -220 <= odds <= 160:
        return "Conservative value"
    if edge < -0.025:
        return "Overvalued"
    if odds >= 190:
        return "High variance"
    return "Neutral"

def context_score(market, odds, point=None):
    score = 0

    # Prefer moderate prices over mega-favorites or longshots
    if -170 <= odds <= -115:
        score += 1
    if 100 <= odds <= 145:
        score += 1
    if odds < -230:
        score -= 2
    if odds > 180:
        score -= 2

    # Game-market heuristics
    if market == "h2h":
        score += 1

    if market == "spreads":
        # +1.5 run line can be lower variance but often juiced
        if point == 1.5 or point == -1.5:
            score += 0

    if market == "totals":
        # totals are more sensitive to weather/stadium/lineups, so keep them slightly more cautious
        score -= 1

    return score

def projection_from_context(implied_probability, odds, market, point=None):
    projected = implied_probability

    # Price-shape adjustment
    if odds < -250:
        projected -= 0.035
    elif -250 <= odds < -190:
        projected -= 0.015
    elif -180 <= odds <= -120:
        projected += 0.012
    elif 100 <= odds <= 150:
        projected += 0.008
    elif odds >= 200:
        projected -= 0.035

    projected += context_score(market, odds, point) * 0.008

    return max(0.01, min(0.99, projected))

def fetch_odds(api_key):
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

def extract_sportsbooks(raw):
    return sorted({
        b.get("title", "")
        for g in raw
        for b in g.get("bookmakers", [])
        if b.get("title")
    })

def analyze(raw, sportsbook_filter="Best available"):
    rows = []

    for game in raw:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_name = f"{away} @ {home}"

        books = game.get("bookmakers", [])
        if sportsbook_filter and sportsbook_filter != "Best available":
            books = [b for b in books if b.get("title") == sportsbook_filter]

        for book in books:
            book_name = book.get("title", "")

            for market in book.get("markets", []):
                market_key = market.get("key", "")

                for outcome in market.get("outcomes", []):
                    odds = outcome.get("price")
                    if odds is None:
                        continue

                    label = outcome.get("name", "")
                    point = outcome.get("point", None)

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

                    implied = american_to_implied(odds)
                    projected = projection_from_context(implied, odds, market_key, point)
                    edge = projected - implied
                    decimal = american_to_decimal(odds)
                    ev = expected_value(projected, decimal)

                    rows.append({
                        "game": game_name,
                        "sportsbook": book_name,
                        "market": market_name,
                        "pick": pick,
                        "odds": int(odds),
                        "line": point,
                        "implied_prob": implied,
                        "projected_prob": projected,
                        "edge": edge,
                        "ev_per_$1": ev,
                        "classification": classify(edge, odds, projected),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if sportsbook_filter == "Best available":
        df["decimal"] = df["odds"].apply(american_to_decimal)
        df = df.sort_values("decimal", ascending=False).drop_duplicates(["game", "market", "pick"])
        df = df.drop(columns=["decimal"])

    return df.sort_values(["edge", "ev_per_$1"], ascending=False)

def build_parlays(df, size=2, max_results=10):
    eligible = df[
        (df["classification"].isin(["Best value", "Conservative value"])) &
        (df["odds"] >= -220) &
        (df["odds"] <= 160)
    ].copy()

    rows = []

    for combo in itertools.combinations(eligible.to_dict("records"), size):
        games = [c["game"] for c in combo]

        # Avoid same-game correlation in starter mode
        if len(games) != len(set(games)):
            continue

        decimal_odds = math.prod(american_to_decimal(c["odds"]) for c in combo)
        implied = 1 / decimal_odds
        projected = math.prod(c["projected_prob"] for c in combo)
        edge = projected - implied

        if decimal_odds >= 2:
            american_text = f"+{round((decimal_odds - 1) * 100)}"
        else:
            american_text = str(round(-100 / (decimal_odds - 1)))

        rows.append({
            "parlay": " + ".join([f'{c["market"]}: {c["pick"]} ({c["odds"]:+d})' for c in combo]),
            "games": " | ".join(games),
            "odds": american_text,
            "projected_win_%": round(projected * 100, 1),
            "edge_%": round(edge * 100, 2),
            "note": "Avoids same-game correlation. Verify starters, lineups, weather, and news."
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("edge_%", ascending=False).head(max_results)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("The Odds API key", type="password", value=os.getenv("ODDS_API_KEY", ""))

    market_filter = st.multiselect(
        "Markets",
        ["Moneyline", "Run line", "Total"],
        default=["Moneyline", "Run line", "Total"]
    )

    risk_mode = st.selectbox(
        "Risk filter",
        ["Conservative/value only", "Balanced", "Show everything"],
        index=0
    )

    parlay_size = st.selectbox("Parlay size", [2, 3], index=0)

    run_button = st.button("Analyze MLB board", type="primary")

if run_button:
    if not api_key:
        st.error("Enter your Odds API key first.")
        st.stop()

    with st.spinner("Pulling MLB odds..."):
        try:
            raw = fetch_odds(api_key)
        except Exception as e:
            st.error(f"Could not pull odds. Error: {e}")
            st.stop()

    sportsbook_names = extract_sportsbooks(raw)
    sportsbook_choice = st.selectbox("Sportsbook view", ["Best available"] + sportsbook_names)

    df = analyze(raw, sportsbook_choice)

    if df.empty:
        st.info("No MLB odds found right now.")
        st.stop()

    if market_filter:
        df = df[df["market"].isin(market_filter)]

    if risk_mode == "Conservative/value only":
        display_df = df[df["classification"].isin(["Best value", "Conservative value"])].copy()
    elif risk_mode == "Balanced":
        display_df = df[df["classification"] != "High variance"].copy()
    else:
        display_df = df.copy()

    st.subheader("Best individual bets")

    if display_df.empty:
        st.info("No conservative/value bets found with current filters.")
    else:
        show = display_df.copy()
        show["implied_prob"] = (show["implied_prob"] * 100).round(1).astype(str) + "%"
        show["projected_prob"] = (show["projected_prob"] * 100).round(1).astype(str) + "%"
        show["edge"] = (show["edge"] * 100).round(2).astype(str) + "%"
        show["ev_per_$1"] = show["ev_per_$1"].round(3)

        st.dataframe(
            show[[
                "game", "sportsbook", "market", "pick", "odds", "line",
                "implied_prob", "projected_prob", "edge",
                "ev_per_$1", "classification"
            ]].head(50),
            use_container_width=True,
            hide_index=True
        )

    st.subheader(f"Candidate {parlay_size}-leg parlays")

    parlays = build_parlays(df, size=parlay_size)

    if parlays.empty:
        st.info("No parlays met the conservative filter.")
    else:
        st.dataframe(parlays, use_container_width=True, hide_index=True)

    st.subheader("Overvalued / avoid list")

    avoid = df[df["classification"] == "Overvalued"].copy()

    if avoid.empty:
        st.info("No obviously overvalued prices found by the starter model.")
    else:
        avoid_show = avoid.copy()
        avoid_show["edge"] = (avoid_show["edge"] * 100).round(2).astype(str) + "%"
        st.dataframe(
            avoid_show[["game", "sportsbook", "market", "pick", "odds", "edge"]],
            use_container_width=True,
            hide_index=True
        )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full analysis CSV", csv, "mlb_basic_bet_analysis.csv", "text/csv")

else:
    st.info("Enter your API key in the sidebar, then tap Analyze.")
