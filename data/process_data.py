import pandas as pd
import numpy as np
import ast

print("Loading raw data...")
df = pd.read_csv('data/raw_shots.csv')

# המרה בטוחה של עמודת המיקום ממחרוזת לרשימה
df['location'] = df['location'].apply(ast.literal_eval)

# פיצול לקואורדינטות X ו-Y
df['x'] = df['location'].apply(lambda loc: loc[0])
df['y'] = df['location'].apply(lambda loc: loc[1])

# הסרת פנדלים - מתנהגים סטטיסטית שונה מבעיטות שדה
df = df[df['shot_type'] != 'Penalty']

# חישוב מרחק לשער
df['distance_to_goal'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)

# חישוב זווית לשער
def calculate_angle(x, y):
    if x == 120:
        return 0.0
    # וקטורים אל שתי קורות השער
    v1 = np.array([120 - x, 36 - y])
    v2 = np.array([120 - x, 44 - y])
    
    # חישוב הזווית ביניהם
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

df['angle_to_goal'] = df.apply(lambda row: calculate_angle(row['x'], row['y']), axis=1)

# הגדרת משתנה המטרה - 1 לגול, 0 לכל דבר אחר
df['is_goal'] = (df['shot_outcome'] == 'Goal').astype(int)

# טיפול בעמודת 'תחת לחץ' - אם חסר ערך, נניח שהשחקן לא היה תחת לחץ
df['under_pressure'] = df['under_pressure'].fillna(False).astype(int)

# בחירת העמודות הרלוונטיות בלבד למודל
features = [
    'is_goal', 'distance_to_goal', 'angle_to_goal', 
    'shot_body_part', 'play_pattern', 'under_pressure', 
    'shot_statsbomb_xg' # שומרים את ה-xG המקורי כדי שנוכל להשוות בסוף
]

df_processed = df[features].copy()

# הסרת שורות עם מידע חסר (Drop NA)
df_processed = df_processed.dropna()

# שמירה לקובץ חדש ומעובד
output_path = 'data/shots_processed.csv'
df_processed.to_csv(output_path, index=False)
print(f"Feature engineering complete. Data saved to {output_path}")