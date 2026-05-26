import os
import math
import datetime as dt
from functools import lru_cache

import pandas as pd
import requests
import streamlit as st

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="MLB Auto Prop Model",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORT = "baseball_mlb"
REGION = "us"
ODDS_FORMAT = "american"
GAME_MARKETS = ["h2h", "spreads", "totals"]

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .big-card {
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        color: #f8fafc;
        border: 1px solid #334155;
        padding: 1.1rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
        margin-bottom: 1rem;
    }
    .soft-card {
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 18px;
        margin-bottom: .8rem;
    }
    .muted {
        color: #94a3b8;
        font-size: .92rem;
    }
    .green {
        color: #22c55e;
        font-weight: 700;
    }
    .yellow {
        color: #f59e0b;
        font-weight: 700;
    }
    .red {
        color: #ef4444;
        font-weight: 700;
    }
    .pill {
        display: inline-block;
        padding: .25rem .55rem;
        border-radius: 999px;
        background: #1e293b;
        border: 1px solid #475569;
        color: #e2e8f0;
        font-size: .82rem;
        margin-right: .25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MATH HELPERS
# ============================================================

def clamp(x, low, high):
    return max(low, min(high, x))

def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)

def american_to_implied(odds):
    return 1 / american_to_decimal(odds)

def probability_to_fair_american(prob):
    prob = clamp(prob, 0.01, 0.99)
    if prob >= 0.5:
        return int(round(-(prob / (1 - prob)) * 100))
    return int(round(((1 - prob) / prob) * 100))

def poisson_over_probability(mean, line):
    threshold = math.floor(line) + 1
    cumulative = 0.0
    for k in range(threshold):
        cumulative += math.exp(-mean) * (mean ** k) / math.factorial(k)
    return clamp(1 - cumulative, 0.01, 0.99)

def poisson_under_probability(mean, line):
    upper = math.ceil(line) - 1
    cumulative = 0.0
    for k in range(max(0, upper) + 1):
        cumulative += math.exp(-mean) * (mean ** k) / math.factorial(k)
    return clamp(cumulative, 0.01, 0.99)

def fair_odds_note(fair_odds):
    if fair_odds < 0:
        return f"Consider only if sportsbook price is better than {fair_odds}, meaning less juice like -130 instead of -160."
    return f"Consider only if sportsbook price is better than +{fair_odds}, like +170 instead of +140."

def confidence_label(prob, prop_type):
    if prop_type == "Home Run":
        if prob >= 0.16:
            return "Strong HR longshot"
        if prob >= 0.11:
            return "Playable HR lean"
        return "Low-probability HR"
    if prob >= 0.68:
        return "Strong projection"
    if prob >= 0.58:
        return "Good projection"
    if prob >= 0.52:
        return "Small lean"
    return "Pass / thin"

# ============================================================
# MLB API HELPERS
# ============================================================

@st.cache_data(ttl=60 * 60 * 24)
def search_player(name):
    url = f"{MLB_API}/people/search"
    r = requests.get(url, params={"names": name}, timeout=20)
    r.raise_for_status()
    people = r.json().get("people", [])
    return people

@st.cache_data(ttl=60 * 30)
def get_schedule(date_str=None):
    if date_str is None:
        date_str = dt.date.today().isoformat()
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team,linescore"
    }
    r = requests.get(f"{MLB_API}/schedule", params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def find_player_game(player_team_id, schedule_json):
    for d in schedule_json.get("dates", []):
        for game in d.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {})
            away = game.get("teams", {}).get("away", {}).get("team", {})
            if home.get("id") == player_team_id or away.get("id") == player_team_id:
                return game
    return None

@st.cache_data(ttl=60 * 60)
def get_player_season_stats(player_id, group, season):
    """
    group: hitting or pitching
    """
    url = f"{MLB_API}/people/{player_id}/stats"
    params = {
        "stats": "season",
        "group": group,
        "season": season
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {})

@st.cache_data(ttl=60 * 60)
def get_player_recent_game_log(player_id, group, season):
    url = f"{MLB_API}/people/{player_id}/stats"
    params = {
        "stats": "gameLog",
        "group": group,
        "season": season
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    return splits

@st.cache_data(ttl=60 * 60)
def get_team_roster(team_id):
    url = f"{MLB_API}/teams/{team_id}/roster"
    r = requests.get(url, params={"rosterType": "active"}, timeout=20)
    r.raise_for_status()
    return r.json().get("roster", [])

def safe_float(x, default=0.0):
    try:
        if x in [None, "", "-", ".---"]:
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x in [None, "", "-"]:
            return default
        return int(float(x))
    except Exception:
        return default

def get_current_season():
    today = dt.date.today()
    return today.year

def extract_player_team_id(person_obj):
    current_team = person_obj.get("currentTeam", {})
    return current_team.get("id"), current_team.get("name", "")

def probable_pitchers_from_game(game):
    home = game.get("teams", {}).get("home", {})
    away = game.get("teams", {}).get("away", {})
    return {
        "home_team": home.get("team", {}).get("name", ""),
        "home_team_id": home.get("team", {}).get("id"),
        "home_probable": home.get("probablePitcher", {}),
        "away_team": away.get("team", {}).get("name", ""),
        "away_team_id": away.get("team", {}).get("id"),
        "away_probable": away.get("probablePitcher", {}),
    }

# ============================================================
# ODDS API HELPERS
# ============================================================

@st.cache_data(ttl=60 * 10)
def fetch_game_odds(api_key):
    if not api_key:
        return pd.DataFrame()
    url = f"{ODDS_API_BASE}/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": ",".join(GAME_MARKETS),
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    raw = r.json()

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
    return pd.DataFrame(rows)

# ============================================================
# AUTOMATIC FEATURE ENGINE
# ============================================================

def recent_hitting_features(game_log, n=10):
    recent = game_log[-n:] if len(game_log) >= n else game_log
    if not recent:
        return {
            "recent_games": 0,
            "hit_rate": 0.0,
            "avg_hits": 0.0,
            "avg_total_bases": 0.0,
            "avg_rbi": 0.0,
            "avg_runs": 0.0,
            "hr_rate": 0.0,
        }

    hits = []
    tb = []
    rbi = []
    runs = []
    hr = []

    for g in recent:
        stat = g.get("stat", {})
        h = safe_int(stat.get("hits"))
        total_bases = safe_int(stat.get("totalBases"))
        rb = safe_int(stat.get("rbi"))
        rr = safe_int(stat.get("runs"))
        hh = safe_int(stat.get("homeRuns"))
        hits.append(h)
        tb.append(total_bases)
        rbi.append(rb)
        runs.append(rr)
        hr.append(hh)

    return {
        "recent_games": len(recent),
        "hit_rate": sum(1 for x in hits if x >= 1) / len(hits),
        "avg_hits": sum(hits) / len(hits),
        "avg_total_bases": sum(tb) / len(tb),
        "avg_rbi": sum(rbi) / len(rbi),
        "avg_runs": sum(runs) / len(runs),
        "hr_rate": sum(1 for x in hr if x >= 1) / len(hr),
    }

def recent_pitching_features(game_log, n=5):
    recent = game_log[-n:] if len(game_log) >= n else game_log
    if not recent:
        return {
            "recent_games": 0,
            "avg_ks": 0.0,
            "avg_outs": 0.0,
            "avg_er": 0.0,
        }

    ks, outs, er = [], [], []
    for g in recent:
        stat = g.get("stat", {})
        ks.append(safe_int(stat.get("strikeOuts")))
        er.append(safe_int(stat.get("earnedRuns")))

        ip = str(stat.get("inningsPitched", "0"))
        # MLB IP format 5.1 = 5 innings + 1 out, 5.2 = 5 innings + 2 outs.
        if "." in ip:
            whole, frac = ip.split(".")
            outs.append(safe_int(whole) * 3 + safe_int(frac))
        else:
            outs.append(safe_int(ip) * 3)

    return {
        "recent_games": len(recent),
        "avg_ks": sum(ks) / len(ks),
        "avg_outs": sum(outs) / len(outs),
        "avg_er": sum(er) / len(er),
    }

def season_hitting_rates(stats):
    games = max(1, safe_int(stats.get("gamesPlayed"), 1))
    hits = safe_int(stats.get("hits"))
    tb = safe_int(stats.get("totalBases"))
    rbi = safe_int(stats.get("rbi"))
    runs = safe_int(stats.get("runs"))
    hr = safe_int(stats.get("homeRuns"))
    avg = safe_float(stats.get("avg"))
    ops = safe_float(stats.get("ops"))

    return {
        "games": games,
        "hits_per_game": hits / games,
        "tb_per_game": tb / games,
        "rbi_per_game": rbi / games,
        "runs_per_game": runs / games,
        "hr_per_game": hr / games,
        "avg": avg,
        "ops": ops,
    }

def season_pitching_rates(stats):
    games = max(1, safe_int(stats.get("gamesPlayed"), 1))
    starts = max(1, safe_int(stats.get("gamesStarted"), games))
    ks = safe_int(stats.get("strikeOuts"))
    er = safe_int(stats.get("earnedRuns"))
    innings = safe_float(stats.get("inningsPitched"))

    outs = innings * 3

    return {
        "games": games,
        "starts": starts,
        "ks_per_start": ks / starts,
        "outs_per_start": outs / starts,
        "er_per_start": er / starts,
        "era": safe_float(stats.get("era")),
        "whip": safe_float(stats.get("whip")),
        "k9": safe_float(stats.get("strikeoutsPer9Inn")),
    }

def opponent_pitcher_quality(probable_pitcher_stats):
    if not probable_pitcher_stats:
        return 0.0
    era = safe_float(probable_pitcher_stats.get("era"), 4.25)
    whip = safe_float(probable_pitcher_stats.get("whip"), 1.30)
    k9 = safe_float(probable_pitcher_stats.get("strikeoutsPer9Inn"), 8.5)

    score = 0.0
    score += (4.25 - era) * 0.12
    score += (1.30 - whip) * 0.70
    score += (k9 - 8.5) * 0.05
    return clamp(score, -1.0, 1.0)

# ============================================================
# PROJECTION MODELS
# ============================================================

def project_hitter_prop(prop, line, season_rates, recent_features, opposing_pitcher_quality_score, team_total=4.3):
    notes = []

    # Blend season and recent rates.
    if prop == "Hit Over":
        mean = season_rates["hits_per_game"] * 0.65 + recent_features["avg_hits"] * 0.35
        mean += (season_rates["avg"] - 0.250) * 1.1
        mean += (team_total - 4.3) * 0.045
        mean -= opposing_pitcher_quality_score * 0.12
        prob = poisson_over_probability(max(0.15, mean), line)

    elif prop == "Total Bases Over":
        mean = season_rates["tb_per_game"] * 0.60 + recent_features["avg_total_bases"] * 0.40
        mean += (season_rates["ops"] - 0.720) * 0.55
        mean += (team_total - 4.3) * 0.070
        mean -= opposing_pitcher_quality_score * 0.16
        prob = poisson_over_probability(max(0.15, mean), line)

    elif prop == "RBI Over":
        mean = season_rates["rbi_per_game"] * 0.70 + recent_features["avg_rbi"] * 0.30
        mean += (team_total - 4.3) * 0.095
        mean -= opposing_pitcher_quality_score * 0.08
        prob = poisson_over_probability(max(0.05, mean), line)

    elif prop == "Run Over":
        mean = season_rates["runs_per_game"] * 0.70 + recent_features["avg_runs"] * 0.30
        mean += (team_total - 4.3) * 0.090
        mean -= opposing_pitcher_quality_score * 0.06
        prob = poisson_over_probability(max(0.05, mean), line)

    elif prop == "Home Run":
        base = season_rates["hr_per_game"] * 0.70 + recent_features["hr_rate"] * 0.30
        base += (season_rates["ops"] - 0.720) * 0.055
        base += (team_total - 4.3) * 0.012
        base -= opposing_pitcher_quality_score * 0.018
        prob = clamp(base, 0.005, 0.33)

    else:
        prob = 0.50

    if recent_features["recent_games"] > 0:
        notes.append(f"recent sample: last {recent_features['recent_games']} games")
    if recent_features.get("hit_rate", 0) >= 0.70 and prop == "Hit Over":
        notes.append("strong recent hit rate")
    if season_rates.get("ops", 0) >= 0.800 and prop in ["Total Bases Over", "Home Run"]:
        notes.append("above-average power/on-base profile")
    if opposing_pitcher_quality_score > 0.35:
        notes.append("opposing pitcher suppresses projection")
    elif opposing_pitcher_quality_score < -0.35:
        notes.append("weaker opposing pitcher boosts projection")
    if team_total > 4.7:
        notes.append("positive team scoring context")

    return clamp(prob, 0.01, 0.99), notes

def project_pitcher_prop(prop, line, season_rates, recent_features, opponent_team_name="Opponent"):
    notes = []

    if prop == "Strikeouts Over":
        mean = season_rates["ks_per_start"] * 0.65 + recent_features["avg_ks"] * 0.35
        if season_rates.get("k9", 0) >= 9.0:
            mean += 0.25
            notes.append("strong K/9 profile")
        prob = poisson_over_probability(max(1.0, mean), line)

    elif prop == "Outs Recorded Over":
        mean = season_rates["outs_per_start"] * 0.65 + recent_features["avg_outs"] * 0.35
        if season_rates.get("era", 4.5) <= 3.50:
            mean += 0.6
            notes.append("run prevention supports length")
        if season_rates.get("whip", 1.4) <= 1.15:
            mean += 0.4
            notes.append("low WHIP supports efficiency")
        prob = poisson_over_probability(max(6.0, mean), line)

    elif prop == "Earned Runs Under":
        mean = season_rates["er_per_start"] * 0.65 + recent_features["avg_er"] * 0.35
        if season_rates.get("era", 4.5) <= 3.50:
            mean -= 0.20
            notes.append("strong ERA profile")
        if season_rates.get("whip", 1.4) <= 1.15:
            mean -= 0.15
            notes.append("low WHIP profile")
        prob = poisson_under_probability(max(0.35, mean), line)

    else:
        prob = 0.50

    if recent_features["recent_games"] > 0:
        notes.append(f"recent sample: last {recent_features['recent_games']} starts/appearances")
    notes.append(f"opponent: {opponent_team_name}")

    return clamp(prob, 0.01, 0.99), notes

# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="big-card">
        <h1>⚾ MLB Auto Prop Model</h1>
        <div class="muted">
            Enter a player, prop type, and line. The app automatically pulls MLB season stats, recent game logs,
            schedule context, probable-pitcher context when available, then returns projected probability and fair odds.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "No bet is guaranteed. This model estimates fair price. You still need to compare the fair odds to your sportsbook's actual prop odds."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Optional Odds API key", type="password", value=os.getenv("ODDS_API_KEY", ""))
    st.caption("Used only for game odds context. Player prop projections work without prop odds.")
    season = st.number_input("Season", min_value=2021, max_value=dt.date.today().year, value=get_current_season(), step=1)
    date_str = st.date_input("Game date", value=dt.date.today()).isoformat()

tab1, tab2, tab3 = st.tabs(["Auto player prop", "Game odds", "Model notes"])

with tab1:
    col1, col2, col3 = st.columns([1.2, 1, .8])

    with col1:
        player_name = st.text_input("Player name", placeholder="Example: Aaron Judge, Juan Soto, Zack Wheeler")

    with col2:
        player_type = st.selectbox("Player type", ["Hitter", "Pitcher"])

    with col3:
        if player_type == "Hitter":
            prop_type = st.selectbox("Prop", ["Hit Over", "Total Bases Over", "RBI Over", "Run Over", "Home Run"])
        else:
            prop_type = st.selectbox("Prop", ["Strikeouts Over", "Outs Recorded Over", "Earned Runs Under"])

    default_line = 0.5
    if prop_type == "Total Bases Over":
        default_line = 1.5
    if prop_type == "Strikeouts Over":
        default_line = 5.5
    if prop_type == "Outs Recorded Over":
        default_line = 17.5
    if prop_type == "Earned Runs Under":
        default_line = 2.5

    line = st.number_input("Prop line", value=float(default_line), step=0.5)

    team_total = 4.3
    if player_type == "Hitter":
        team_total = st.slider(
            "Estimated team total, optional but useful",
            2.0, 7.5, 4.3, 0.1,
            help="If you do not know this, leave it near 4.3. Higher team totals boost hitter props."
        )

    run = st.button("Project prop", type="primary", use_container_width=True)

    if run:
        if not player_name.strip():
            st.error("Enter a player name.")
            st.stop()

        with st.spinner("Searching player and pulling MLB data..."):
            try:
                matches = search_player(player_name.strip())
            except Exception as e:
                st.error(f"Could not search player: {e}")
                st.stop()

        if not matches:
            st.error("No player found. Try full name.")
            st.stop()

        # Pick first search result.
        person = matches[0]
        player_id = person.get("id")
        full_name = person.get("fullName", player_name)
        team_id, team_name = extract_player_team_id(person)

        st.markdown(
            f"""
            <div class="soft-card">
                <h3>{full_name}</h3>
                <span class="pill">{player_type}</span>
                <span class="pill">{team_name or "Team unknown"}</span>
                <span class="pill">{prop_type} {line:g}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        group = "hitting" if player_type == "Hitter" else "pitching"

        try:
            season_stats = get_player_season_stats(player_id, group, int(season))
            logs = get_player_recent_game_log(player_id, group, int(season))
        except Exception as e:
            st.error(f"Could not pull player stats: {e}")
            st.stop()

        try:
            sched = get_schedule(date_str)
            game = find_player_game(team_id, sched) if team_id else None
        except Exception:
            game = None

        opposing_pitcher_quality_score = 0.0
        opponent_team_name = "Opponent"
        game_context_text = "No game/probable-pitcher context found for selected date."

        if game:
            pp = probable_pitchers_from_game(game)
            is_home = pp.get("home_team_id") == team_id
            opponent_team_name = pp["away_team"] if is_home else pp["home_team"]
            opponent_probable = pp["away_probable"] if is_home else pp["home_probable"]
            own_probable = pp["home_probable"] if is_home else pp["away_probable"]

            if player_type == "Hitter" and opponent_probable.get("id"):
                try:
                    opp_stats = get_player_season_stats(opponent_probable["id"], "pitching", int(season))
                    opposing_pitcher_quality_score = opponent_pitcher_quality(opp_stats)
                    game_context_text = f"Opponent: {opponent_team_name}. Probable pitcher: {opponent_probable.get('fullName', 'Unknown')}."
                except Exception:
                    game_context_text = f"Opponent: {opponent_team_name}. Probable pitcher found, stats unavailable."
            elif player_type == "Pitcher":
                game_context_text = f"Opponent: {opponent_team_name}. Schedule context found."

        if player_type == "Hitter":
            sr = season_hitting_rates(season_stats)
            rf = recent_hitting_features(logs, 10)
            prob, notes = project_hitter_prop(prop_type, line, sr, rf, opposing_pitcher_quality_score, team_total=team_total)
        else:
            sr = season_pitching_rates(season_stats)
            rf = recent_pitching_features(logs, 5)
            prob, notes = project_pitcher_prop(prop_type, line, sr, rf, opponent_team_name=opponent_team_name)

        fair_odds = probability_to_fair_american(prob)
        confidence = confidence_label(prob, prop_type)

        m1, m2, m3 = st.columns(3)
        m1.metric("Projected probability", f"{prob * 100:.1f}%")
        m2.metric("Fair odds", f"{fair_odds:+d}")
        m3.metric("Confidence", confidence)

        st.markdown(
            f"""
            <div class="soft-card">
                <b>How to use this:</b><br>
                <span class="green">{fair_odds_note(fair_odds)}</span><br><br>
                <span class="muted">{game_context_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        reason_rows = []
        for n in notes:
            reason_rows.append({"reason": n})

        if reason_rows:
            st.subheader("Why the model says this")
            st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)

        st.subheader("Underlying automatic data")

        if player_type == "Hitter":
            data = {
                "season_games": sr["games"],
                "hits_per_game": round(sr["hits_per_game"], 3),
                "total_bases_per_game": round(sr["tb_per_game"], 3),
                "rbi_per_game": round(sr["rbi_per_game"], 3),
                "runs_per_game": round(sr["runs_per_game"], 3),
                "hr_per_game": round(sr["hr_per_game"], 3),
                "avg": sr["avg"],
                "ops": sr["ops"],
                "recent_hit_rate": round(rf["hit_rate"], 3),
                "recent_avg_hits": round(rf["avg_hits"], 3),
                "recent_avg_total_bases": round(rf["avg_total_bases"], 3),
                "recent_hr_rate": round(rf["hr_rate"], 3),
                "opposing_pitcher_quality_score": round(opposing_pitcher_quality_score, 3),
                "team_total_used": team_total,
            }
        else:
            data = {
                "season_games": sr["games"],
                "season_starts": sr["starts"],
                "ks_per_start": round(sr["ks_per_start"], 3),
                "outs_per_start": round(sr["outs_per_start"], 3),
                "er_per_start": round(sr["er_per_start"], 3),
                "era": sr["era"],
                "whip": sr["whip"],
                "k9": sr["k9"],
                "recent_avg_ks": round(rf["avg_ks"], 3),
                "recent_avg_outs": round(rf["avg_outs"], 3),
                "recent_avg_er": round(rf["avg_er"], 3),
            }

        st.dataframe(pd.DataFrame([data]), use_container_width=True, hide_index=True)

        export = pd.DataFrame([{
            "player": full_name,
            "team": team_name,
            "player_type": player_type,
            "prop": prop_type,
            "line": line,
            "projected_probability": round(prob, 4),
            "fair_odds": fair_odds,
            "confidence": confidence,
            "game_context": game_context_text,
            "notes": "; ".join(notes),
            **data
        }])

        st.download_button(
            "Download projection CSV",
            export.to_csv(index=False).encode("utf-8"),
            f"{full_name.replace(' ', '_').lower()}_prop_projection.csv",
            "text/csv",
            use_container_width=True
        )

with tab2:
    st.subheader("Game odds scanner")

    if not api_key:
        st.info("Add your Odds API key in Settings to use game odds. This is optional for prop projections.")
    else:
        with st.spinner("Pulling game odds..."):
            try:
                odds_df = fetch_game_odds(api_key)
            except Exception as e:
                st.error(f"Could not pull odds: {e}")
                odds_df = pd.DataFrame()

        if not odds_df.empty:
            books = ["Best available"] + sorted(odds_df["sportsbook"].dropna().unique().tolist())
            selected_book = st.selectbox("Sportsbook", books)

            if selected_book == "Best available":
                odds_df["decimal"] = odds_df["odds"].apply(american_to_decimal)
                view = odds_df.sort_values("decimal", ascending=False).drop_duplicates(["game", "market", "pick"])
            else:
                view = odds_df[odds_df["sportsbook"] == selected_book].copy()

            view["implied_prob"] = (view["implied_prob"] * 100).round(1).astype(str) + "%"
            st.dataframe(
                view[["game", "sportsbook", "market", "pick", "odds", "line", "implied_prob"]],
                use_container_width=True,
                hide_index=True
            )

with tab3:
    st.subheader("Model notes")

    st.markdown(
        """
        This version removes the manual sliders from the main workflow.

        **You enter:**
        - player name
        - hitter or pitcher
        - prop type
        - prop line

        **The app automatically pulls:**
        - MLB player identity
        - current team
        - season hitting/pitching stats
        - recent game logs
        - today's schedule
        - opponent and probable pitcher when available
        - optional game odds if you provide an Odds API key

        **It outputs:**
        - projected probability
        - fair American odds
        - confidence tier
        - reasoning
        - underlying data table

        **Still missing from this free-data version:**
        - live confirmed lineups
        - paid prop odds
        - Baseball Savant batted-ball hit charts
        - weather/wind API
        - umpire assignments
        - paid injury/news feed

        The model is meant to be a fair-price assistant, not a lock machine.
        """
    )
