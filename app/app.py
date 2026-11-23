from flask import Flask, request, jsonify
import psycopg2
import json
import os
import pickle

app = Flask(__name__)

# Загрузка модели при старте
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "10.0.2.101"),
        dbname=os.getenv("DB_NAME", "ml_predictions"),
        user=os.getenv("DB_USER", "ml_app"),
        password=os.getenv("DB_PASSWORD", "Ds_22112003"),
        sslmode=os.getenv("DB_SSLMODE", "require")
    )

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "No features provided"}), 400

    # Ожидается 8 чисел: pregnancies, glucose, blood_pressure,
    # skin_thickness, insulin, bmi, diabetes_pedigree, age
    pred = model.predict([features])[0]
    proba = max(model.predict_proba([features])[0])
    result = {"class": int(pred), "probability": float(proba)}

    # Запись в БД
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (input_data, prediction, model_version) VALUES (%s, %s, %s)",
            (json.dumps({"features": features}), json.dumps(result), "v1.0-diabetes")
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        # Если БД недоступна, возвращаем предсказание, но предупреждаем об ошибке БД
        # (для отладки выводим ошибку в логи контейнера)
        print(f"Database error: {e}")
        return jsonify({"error": f"DB error: {str(e)}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
