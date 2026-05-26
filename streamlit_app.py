import os
import math
import itertools
import requests
import pandas as pd
import streamlit as st

SPORT = "baseball_mlb"
REGION = "us"
MARKETS = "h2h"
ODDS_FORMAT = "american"
BASE_URL = "https://api.the-odds-api.com/v4/sports"


st.set_page_config(
    page_title="MLB Bet Analyzer",
    page_icon="⚾",
    layout="centered"
)

st.title("⚾ MLB Bet Analyzer")
st.caption("Mobile-friendly odds scanner for conservative/value MLB bets.")

st.warning(
    "No bet is guaranteed or truly safe. This tool is for analysis only. "
    "Use small units and verify lineups, pitchers, injuries, and weather before betting."
)


def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def american_to_implied(odds):
    return 1 / american_to_decimal(odds)


def expected_value(probability, decimal_odds):
    return probability * (decimal_odds - 1) - (1 - probability)


def simple_projection_model(implied_probability, odds):
    """
    Starter model.
    It estimates where prices may be conservative/value/overvalued.
    Upgrade later with pitchers, bullpen, injuries, weather, and lineups.
    """
    projected = implied_probability

    if odds < -250:
        projected -= 0.035
    elif -250 <= odds < -190:
        projected -= 0.015
    elif -180 <= odds <= -120:
        projected += 0.020
    elif 100 <= odds <= 150:
        projected += 0.010
    elif odds >= 200:
        projected -= 0.040

    return max(0.01, min(0.99, projected))


def classify(edge, odds):
    if edge >= 0.04 and -180 <= odds <= 140:
        return "Strong value"
    if edge >= 0.02 and -220 <= odds <= 150:
        return "Conservative value"
    if edge < -0.025:
        return "Likely overvalued"
    if odds >= 180:
        return "High variance"
    return "Neutral"


def fetch_odds(api_key):
    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def analyze(data, sportsbook_filter=None):
    rows = []

    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_name = f"{away} @ {home}"

        books = game.get("bookmakers", [])
        if sportsbook_filter and sportsbook_filter != "Best available":
            books = [b for b in books if b.get("title") == sportsbook_filter]

        for book in books:
            book_name = book.get("title", "")
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    pick = outcome.get("name", "")
                    odds = outcome.get("price")

                    if odds is None:
                        continue

                    implied = american_to_implied(odds)
                    projected = simple_projection_model(implied, odds)
                    edge = projected - implied
                    decimal = american_to_decimal(odds)
                    ev = expected_value(projected, decimal)

                    rows.append({
                        "game": game_name,
                        "sportsbook": book_name,
                        "pick": pick,
                        "odds": int(odds),
                        "implied_prob": implied,
                        "projected_prob": projected,
                        "edge": edge,
                        "ev_per_$1": ev,
                        "classification": classify(edge, odds),
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if sportsbook_filter == "Best available":
        # Keep best odds for each pick across books
        df["decimal"] = df["odds"].apply(american_to_decimal)
        df = df.sort_values("decimal", ascending=False).drop_duplicates(["game", "pick"])
        df = df.drop(columns=["decimal"])

    return df.sort_values(["edge", "ev_per_$1"], ascending=False)


def build_parlays(df, size=2, max_results=10):
    eligible = df[
        (df["classification"].isin(["Strong value", "Conservative value"])) &
        (df["odds"] >= -220) &
        (df["odds"] <= 150)
    ].copy()

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
            american = round((decimal_odds - 1) * 100)
            american_text = f"+{american}"
        else:
            american = round(-100 / (decimal_odds - 1))
            american_text = str(american)

        rows.append({
            "parlay": " + ".join([f'{c["pick"]} ({c["odds"]:+d})' for c in combo]),
            "games": " | ".join(games),
            "odds": american_text,
            "projected_win_%": round(projected * 100, 1),
            "edge_%": round(edge * 100, 2),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("edge_%", ascending=False).head(max_results)


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "The Odds API key",
        type="password",
        value=os.getenv("ODDS_API_KEY", "")
    )

    risk_mode = st.selectbox(
        "Bet style",
        ["Conservative", "Balanced", "Show everything"]
    )

    parlay_size = st.selectbox("Parlay size", [2, 3], index=0)

    run_button = st.button("Analyze today's MLB board", type="primary")


if run_button:
    if not api_key:
        st.error("Enter your Odds API key first.")
        st.stop()

    with st.spinner("Pulling live MLB odds..."):
        try:
            raw = fetch_odds(api_key)
        except Exception as e:
            st.error(f"Could not pull odds: {e}")
            st.stop()

    sportsbook_names = sorted({
        b.get("title", "")
        for g in raw
        for b in g.get("bookmakers", [])
        if b.get("title")
    })

    sportsbook_choice = st.selectbox(
        "Sportsbook view",
        ["Best available"] + sportsbook_names
    )

    df = analyze(raw, sportsbook_choice)

    if df.empty:
        st.info("No MLB odds found right now.")
        st.stop()

    if risk_mode == "Conservative":
        display_df = df[df["classification"].isin(["Strong value", "Conservative value"])].copy()
    elif risk_mode == "Balanced":
        display_df = df[df["classification"] != "High variance"].copy()
    else:
        display_df = df.copy()

    st.subheader("Best individual bets")
    if display_df.empty:
        st.info("No conservative/value bets found with the current filters.")
    else:
        show = display_df.copy()
        show["implied_prob"] = (show["implied_prob"] * 100).round(1).astype(str) + "%"
        show["projected_prob"] = (show["projected_prob"] * 100).round(1).astype(str) + "%"
        show["edge"] = (show["edge"] * 100).round(2).astype(str) + "%"
        show["ev_per_$1"] = show["ev_per_$1"].round(3)
        st.dataframe(
            show[[
                "game", "sportsbook", "pick", "odds",
                "implied_prob", "projected_prob", "edge",
                "ev_per_$1", "classification"
            ]].head(25),
            use_container_width=True,
            hide_index=True
        )

    st.subheader(f"Candidate {parlay_size}-leg parlays")
    parlays = build_parlays(df, size=parlay_size)

    if parlays.empty:
        st.info("No parlays met the conservative filter.")
    else:
        st.dataframe(parlays, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full analysis CSV",
        csv,
        "mlb_bet_analysis.csv",
        "text/csv"
    )

else:
    st.info("Add your API key in the sidebar, then tap Analyze.")
