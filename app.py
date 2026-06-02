import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import ast

st.set_page_config(page_title="xG Calculator", page_icon="⚽", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("model/xg_model.pkl")
    features = joblib.load("model/model_features.pkl")
    return model, features

@st.cache_data
def load_shot_data():
    df = pd.read_csv("data/raw_shots.csv")
    df["location"] = df["location"].apply(ast.literal_eval)
    df["x"] = df["location"].apply(lambda loc: loc[0])
    df["y"] = df["location"].apply(lambda loc: loc[1])
    df = df[df["shot_type"] != "Penalty"]
    df["is_goal"] = df["shot_outcome"] == "Goal"
    return df

def calculate_angle(x, y):
    if x == 120:
        return 0.0
    v1 = np.array([120 - x, 36 - y])
    v2 = np.array([120 - x, 44 - y])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def predict_xg(model, features, distance, angle, body_part, play_pattern, under_pressure):
    row = {f: 0 for f in features}
    row["distance_to_goal"] = distance
    row["angle_to_goal"] = angle
    row["under_pressure"] = int(under_pressure)

    body_col = f"shot_body_part_{body_part}"
    if body_col in row:
        row[body_col] = 1

    pattern_col = f"play_pattern_{play_pattern}"
    if pattern_col in row:
        row[pattern_col] = 1

    X = pd.DataFrame([row])[features]
    return model.predict_proba(X)[0][1]

model, features = load_model()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("⚽ xG Calculator")
st.markdown("**Expected Goals** – What is the probability of this shot resulting in a goal?")
st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Shot Details")

    x_pos = st.slider("X Position (length)", min_value=60, max_value=120, value=100, step=1)
    y_pos = st.slider("Y Position (width)", min_value=0, max_value=80, value=40, step=1)

    distance = round(np.sqrt((120 - x_pos) ** 2 + (40 - y_pos) ** 2), 1)
    angle = round(calculate_angle(x_pos, y_pos), 1)

    st.metric("Distance to Goal", f"{distance} m")
    st.metric("Angle to Goal", f"{angle}°")

    body_part = st.selectbox("Body Part", ["Right Foot", "Left Foot", "Other"])

    play_pattern = st.selectbox("Play Pattern", [
        "Regular Play", "From Counter", "From Free Kick",
        "From Goal Kick", "From Keeper", "From Kick Off", "From Throw In", "Other"
    ])

    under_pressure = st.checkbox("Under pressure from a defender?")

    xg = predict_xg(model, features, distance, angle, body_part, play_pattern, under_pressure)

    st.divider()
    color = "#00c853" if xg > 0.3 else "#ff6d00" if xg > 0.1 else "#d50000"
    st.markdown(f"""
    <div style='text-align:center; padding:20px; background:#1a1a2e; border-radius:12px;'>
        <p style='color:#aaa; font-size:18px; margin:0;'>Expected Goals (xG)</p>
        <p style='color:{color}; font-size:64px; font-weight:bold; margin:0;'>{xg:.2%}</p>
        <p style='color:#aaa; font-size:14px; margin:0;'>chance of scoring</p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("Shot Position on Pitch")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#22312b", line_color="#c7d5cc", half=True)
    fig, ax = pitch.draw(figsize=(7, 5))
    fig.set_facecolor("#22312b")

    size = max(300, xg * 3000)
    color_dot = "#00c853" if xg > 0.3 else "#ff6d00" if xg > 0.1 else "#d50000"
    pitch.scatter(x_pos, y_pos, s=size, c=color_dot, edgecolors="white",
                  linewidths=2, ax=ax, zorder=5)

    ax.annotate(f"xG: {xg:.2%}", xy=(x_pos, y_pos), xytext=(x_pos - 8, y_pos + 5),
                color="white", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", alpha=0.8))

    st.pyplot(fig)
    plt.close()

# ── Shot map section ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Shot Map — FIFA World Cup 2022")

df = load_shot_data()
players = sorted(df["player"].dropna().unique())
default_idx = players.index("Lionel Andrés Messi Cuccittini") if "Lionel Andrés Messi Cuccittini" in players else 0
selected_player = st.selectbox("Select Player", players, index=default_idx)

player_df = df[df["player"] == selected_player]
goals = player_df[player_df["is_goal"]]
misses = player_df[~player_df["is_goal"]]

pitch2 = Pitch(pitch_type="statsbomb", pitch_color="#22312b", line_color="#c7d5cc")
fig2, ax2 = pitch2.draw(figsize=(10, 6))
fig2.set_facecolor("#22312b")

if len(misses):
    pitch2.scatter(misses.x, misses.y, s=misses["shot_statsbomb_xg"] * 1000,
                   c="red", alpha=0.5, ax=ax2, label="Miss")
if len(goals):
    pitch2.scatter(goals.x, goals.y, s=goals["shot_statsbomb_xg"] * 1000,
                   c="lime", alpha=0.9, edgecolors="black", ax=ax2, label="Goal")

total_xg = player_df["shot_statsbomb_xg"].sum()
total_goals = goals.shape[0]
ax2.set_title(f"{selected_player}  |  Goals: {total_goals}  |  xG: {total_xg:.2f}",
              color="white", fontsize=14, pad=10)
ax2.legend(loc="upper left", facecolor="#333", labelcolor="white")

st.pyplot(fig2)
plt.close()
