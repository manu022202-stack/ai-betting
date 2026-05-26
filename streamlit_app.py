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

PROP_MARKETS = [
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_home_runs",
    "batter_strikeouts",
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
    "pitcher_outs",
]

GAME_MARKETS = ["h2h", "spreads", "totals"]

st.set_page_config(page_title="MLB Props & Parlay Analyzer", page_icon="⚾", layout="centered")
st.title("⚾ MLB Props & Parlay Analyzer")
st.caption("Mobile-friendly scanner for MLB sides, totals, and player props.")
st.warning(
    "No bet is guaranteed. This tool estimates value/risk and should be used with bankroll discipline. "
    "Always verify confirmed lineups, scratches, weather, and starting pitchers before betting."
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
    if edge >= 0.05 and projected_prob >= 0.56 and -170 <= odds <= 140:
        return "Best value"
    if edge >= 0.03 and projected_prob >= 0.54 and -200 <= odds <= 160:
        return "Conservative value"
    if edge < -0.025:
        return "Overvalued"
    if odds >= 190:
        return "High variance"
    return "Neutral"

def projection_from_context(implied_probability, odds, context_score=0, prop_type="game"):
    projected = implied_probability
    if odds < -250:
        projected -= 0.035
    elif -250 <= odds < -190:
        projected -= 0.015
    elif -180 <= odds <= -120:
        projected += 0.010
    elif 100 <= odds <= 150:
        projected += 0.005
    elif odds >= 200:
        projected -= 0.035
    projected += context_score * 0.008
    if prop_type == "prop":
        projected = 0.50 + (projected - 0.50) * 0.85
    return max(0.01, min(0.99, projected))

def fetch_odds(api_key, markets):
    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": ",".join(markets),
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

def basic_context_score(row):
    score = 0
    odds = row.get("odds", 0)
    market = row.get("market", "")
    if -170 <= odds <= -115:
        score += 1
    if odds < -230:
        score -= 2
    if odds > 180:
        score -= 2
    if "home_runs" in market:
        score -= 3
    if "hits" in market:
        score += 1
    if "pitcher_strikeouts" in market:
        score += 1
    if "rbis" in market or "runs_scored" in market:
        score -= 1
    return score

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
                    description = outcome.get("description", "")
                    pick = f"{description} - {label}" if description else label
                    if point is not None:
                        pick = f"{pick} {point}"
                    prop_type = "prop" if market_key in PROP_MARKETS else "game"
                    implied = american_to_implied(odds)
                    temp_row = {
                        "game": game_name,
                        "sportsbook": book_name,
                        "market": market_key,
                        "pick": pick,
                        "odds": int(odds),
                        "point": point,
                    }
                    context_score = basic_context_score(temp_row)
                    projected = projection_from_context(
                        implied, odds, context_score=context_score, prop_type=prop_type
                    )
                    edge = projected - implied
                    decimal = american_to_decimal(odds)
                    ev = expected_value(projected, decimal)
                    rows.append({
                        "game": game_name,
                        "sportsbook": book_name,
                        "market": market_key,
                        "pick": pick,
                        "odds": int(odds),
                        "line": point,
                        "bet_type": prop_type,
                        "context_score": context_score,
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

def build_parlays(df, size=2, max_results=10, include_props=True):
    eligible = df[
        (df["classification"].isin(["Best value", "Conservative value"])) &
        (df["odds"] >= -220) &
        (df["odds"] <= 160)
    ].copy()
    if not include_props:
        eligible = eligible[eligible["bet_type"] == "game"]
    rows = []
    for combo in itertools.combinations(eligible.to_dict("records"), size):
        games = [c["game"] for c in combo]
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
            "parlay": " + ".join([f'{c["pick"]} ({c["odds"]:+d})' for c in combo]),
            "games": " | ".join(games),
            "odds": american_text,
            "projected_win_%": round(projected * 100, 1),
            "edge_%": round(edge * 100, 2),
            "note": "Avoids same-game correlation. Verify player status/news."
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("edge_%", ascending=False).head(max_results)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("The Odds API key", type="password", value=os.getenv("ODDS_API_KEY", ""))
    market_mode = st.selectbox(
        "Markets to scan",
        ["Game lines only", "Player props only", "Game lines + player props"],
        index=2
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
    if market_mode == "Game lines only":
        markets = GAME_MARKETS
    elif market_mode == "Player props only":
        markets = PROP_MARKETS
    else:
        markets = GAME_MARKETS + PROP_MARKETS
    with st.spinner("Pulling MLB odds and props..."):
        try:
            raw = fetch_odds(api_key, markets)
        except Exception as e:
            st.error(
                "Could not pull odds/props. This can happen if your Odds API plan does not include player props "
                f"or if the market keys are unavailable right now. Error: {e}"
            )
            st.stop()
    sportsbook_names = extract_sportsbooks(raw)
    sportsbook_choice = st.selectbox("Sportsbook view", ["Best available"] + sportsbook_names)
    df = analyze(raw, sportsbook_choice)
    if df.empty:
        st.info("No odds/props found right now.")
        st.stop()
    if risk_mode == "Conservative/value only":
        display_df = df[df["classification"].isin(["Best value", "Conservative value"])].copy()
    elif risk_mode == "Balanced":
        display_df = df[df["classification"] != "High variance"].copy()
    else:
        display_df = df.copy()
    st.subheader("Best individual bets / props")
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
                "game", "sportsbook", "bet_type", "market", "pick", "odds", "line",
                "context_score", "implied_prob", "projected_prob", "edge",
                "ev_per_$1", "classification"
            ]].head(50),
            use_container_width=True,
            hide_index=True
        )
    st.subheader(f"Candidate {parlay_size}-leg parlays")
    parlays = build_parlays(df, size=parlay_size, include_props=True)
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
            avoid_show[["game", "sportsbook", "bet_type", "market", "pick", "odds", "edge"]],
            use_container_width=True,
            hide_index=True
        )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full analysis CSV", csv, "mlb_props_bet_analysis.csv", "text/csv")
else:
    st.info("Enter your API key in the sidebar, then tap Analyze.")
