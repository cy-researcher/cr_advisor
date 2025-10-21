import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("career_data.csv")

# Preprocess
data['skills'] = data['skills'].apply(lambda x: x.split(';'))
data['interests'] = data['interests'].apply(lambda x: x.split(';'))

mlb_skills = MultiLabelBinarizer()
mlb_interests = MultiLabelBinarizer()

skills_encoded = mlb_skills.fit_transform(data['skills'])
interests_encoded = mlb_interests.fit_transform(data['interests'])

edu_encoder = LabelEncoder()
education_encoded = edu_encoder.fit_transform(data['education_level'])

X = pd.concat([
    pd.DataFrame(skills_encoded, columns=mlb_skills.classes_),
    pd.DataFrame(interests_encoded, columns=mlb_interests.classes_),
    pd.Series(education_encoded, name="education_level")
], axis=1)

y = data['career']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "career_model.pkl")
joblib.dump(mlb_skills, "mlb_skills.pkl")
joblib.dump(mlb_interests, "mlb_interests.pkl")
joblib.dump(edu_encoder, "edu_encoder.pkl")

print("✅ Model training completed and saved successfully!")
