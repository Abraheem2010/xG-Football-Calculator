import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

print("Loading processed data for training...")
df = pd.read_csv('data/shots_processed.csv')

# Machine Learning לא יודע לקרוא טקסט (כמו "רגל ימין" או "קרן")
# לכן אנחנו הופכים עמודות קטגוריאליות לעמודות של 0 ו-1 (One-Hot Encoding)
categorical_cols = ['shot_body_part', 'play_pattern']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# הפרדה בין הפיצ'רים (X) למשתנה המטרה - גול או לא גול (y)
# אנחנו מוציאים את ה-xG של StatsBomb מהפיצ'רים, כי אסור למודל שלנו להעתיק מהם!
X = df_encoded.drop(columns=['is_goal', 'shot_statsbomb_xg'])
y = df_encoded['is_goal']

# שומרים את ה-xG של StatsBomb בצד כדי להשוות מולו בסוף
sb_xg = df_encoded['shot_statsbomb_xg']

# חלוקת הנתונים: 80% לאימון (למידה) ו-20% לבדיקה (מבחן)
X_train, X_test, y_train, y_test, sb_xg_train, sb_xg_test = train_test_split(
    X, y, sb_xg, test_size=0.2, random_state=42
)

print("Training Gradient Boosting model (This might take a few seconds)...")
# הגדרת המודל עם פרמטרים סטנדרטיים שעובדים טוב ל-xG
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# מבקשים מהמודל שלנו לתת הסתברויות (xG) על קבוצת המבחן (ה-20% שהוא לא ראה)
y_pred_xg = model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance Evaluation ---")
# במודלי הסתברות (כמו xG), ככל שה-Log Loss קרוב יותר ל-0, כך המודל מדויק יותר
model_log_loss = log_loss(y_test, y_pred_xg)
sb_log_loss = log_loss(y_test, sb_xg_test)

# ככל שה-AUC קרוב יותר ל-1, המודל מבחין טוב יותר בין גולים להחמצות
model_auc = roc_auc_score(y_test, y_pred_xg)
sb_auc = roc_auc_score(y_test, sb_xg_test)

print(f"Our Model   -> Log Loss: {model_log_loss:.4f} | ROC-AUC: {model_auc:.4f}")
print(f"StatsBomb   -> Log Loss: {sb_log_loss:.4f} | ROC-AUC: {sb_auc:.4f}")

# שמירת המודל המאומן לתיקיית model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/xg_model.pkl')

# שומרים גם את רשימת העמודות כדי שהדשבורד בעתיד ידע בדיוק איזה נתונים המודל מצפה לקבל
joblib.dump(list(X.columns), 'model/model_features.pkl')

print("\nSuccess! Model trained and saved to 'model/xg_model.pkl'")