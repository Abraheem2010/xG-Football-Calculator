import pandas as pd
from statsbombpy import sb
import os

# יצירת תיקיית דאטה אם היא לא קיימת
os.makedirs('data', exist_ok=True)

print("Fetching World Cup 2022 matches...")
# שליפת כל המשחקים ממונדיאל 2022
matches = sb.matches(competition_id=43, season_id=106)
print(f"Found {len(matches)} matches.")

all_shots = []

# לולאה שעוברת על כל מזהי המשחקים (match_id)
for match_id in matches['match_id']:
    # שליפת כל האירועים של המשחק הספציפי
    events = sb.events(match_id=match_id)
    
    # סינון: שמירת אירועים מסוג 'בעיטה' בלבד
    shots = events[events['type'] == 'Shot'].copy()
    
    # הוספת הבעיטות של המשחק לרשימה הכללית
    all_shots.append(shots)

# חיבור כל טבלאות הבעיטות לטבלה אחת גדולה
df_shots = pd.concat(all_shots, ignore_index=True)
print(f"Total shots extracted from tournament: {len(df_shots)}")

# שמירת הנתונים הגולמיים לקובץ CSV בתיקיית הנתונים
file_path = 'data/raw_shots.csv'
df_shots.to_csv(file_path, index=False)
print(f"Data saved successfully to {file_path}")