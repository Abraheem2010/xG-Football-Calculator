import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import ast
import os

# יצירת תיקיית תמונות אם היא לא קיימת
os.makedirs('figures', exist_ok=True)

print("Loading raw data for visualization...")
df = pd.read_csv('data/raw_shots.csv')

# חילוץ קואורדינטות (X,Y) מהנתונים הגולמיים
df['location'] = df['location'].apply(ast.literal_eval)
df['x'] = df['location'].apply(lambda loc: loc[0])
df['y'] = df['location'].apply(lambda loc: loc[1])

# סינון פנדלים והגדרת גולים
df = df[df['shot_type'] != 'Penalty']
df['is_goal'] = (df['shot_outcome'] == 'Goal')

print("\n--- Top 5 Clinical Finishers (Overperforming xG) ---")
# חישוב: מי השחקנים שהבקיעו יותר ממה שה-xG שלהם צפה?
player_stats = df.groupby('player').agg(
    goals=('is_goal', 'sum'),
    total_xg=('shot_statsbomb_xg', 'sum')
).reset_index()

# הוספת עמודת ביצועים (גולים בפועל פחות xG צפוי)
player_stats['overperformance'] = player_stats['goals'] - player_stats['total_xg']
top_finishers = player_stats[player_stats['goals'] >= 2].sort_values(by='overperformance', ascending=False).head(5)
print(top_finishers[['player', 'goals', 'total_xg', 'overperformance']].to_string(index=False))

print("\nDrawing beautiful shot map for Lionel Messi...")
# נסנן רק את הבעיטות של שחקן ספציפי כדי שהמפה תהיה קריאה ומרשימה
player_name = 'Lionel Andrés Messi Cuccittini'
player_shots = df[df['player'] == player_name]

goals = player_shots[player_shots['is_goal'] == True]
misses = player_shots[player_shots['is_goal'] == False]

# ציור המגרש עם ספריית mplsoccer
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(10, 7))
fig.set_facecolor('#22312b') # רקע כהה

# ציור החמצות (באדום שקוף) - גודל הנקודה לפי ה-xG
pitch.scatter(misses.x, misses.y, s=misses['shot_statsbomb_xg'] * 1000, 
              c='red', alpha=0.5, ax=ax, label='Miss')

# ציור גולים (בירוק בולט) - גודל הנקודה לפי ה-xG
pitch.scatter(goals.x, goals.y, s=goals['shot_statsbomb_xg'] * 1000, 
              c='lime', alpha=0.9, edgecolors='black', ax=ax, label='Goal')

# כותרות ומקרא
plt.title(f"{player_name} - World Cup 2022 Shot Map", fontsize=20, color='white')
ax.legend(loc='upper left')

# שמירת התמונה לתיקיית figures
output_path = 'figures/messi_shot_map.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Shot map successfully saved to {output_path}!")