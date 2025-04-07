import os
import threading
import numpy as np
import tensorflow as tf
import joblib
import time
import json
import serial
import random
import datetime
from datetime import timedelta
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

# Disable TensorFlow OneDNN optimization for compatibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load Forecast Model and Scaler
forecast_model = tf.keras.models.load_model("forecast_lstm_model.keras")
forecast_scaler = joblib.load("forecast_scaler.pkl")

# Serial Configuration
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"Listening on {SERIAL_PORT} at {BAUD_RATE} baud...")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

# Flask & SQLAlchemy Setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic_data.db'
db = SQLAlchemy(app)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")

# Define Database Model
class TrafficData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pm2_5 = db.Column(db.Float, nullable=False)
    pm10 = db.Column(db.Float, nullable=False)
    noise_level = db.Column(db.Float, nullable=False)
    co = db.Column(db.Float, nullable=False)
    no2 = db.Column(db.Float, nullable=False)
    so2 = db.Column(db.Float, nullable=False)
    aqi = db.Column(db.String(20), nullable=False)
    congestion = db.Column(db.Float, nullable=False)  # First predicted congestion value
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class DailyRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pm2_5 = db.Column(db.Float, nullable=False)
    pm10 = db.Column(db.Float, nullable=False)
    noise_level = db.Column(db.Float, nullable=False)
    co = db.Column(db.Float, nullable=False)
    no2 = db.Column(db.Float, nullable=False)
    so2 = db.Column(db.Float, nullable=False)

    aqi = db.Column(db.String(20), nullable=False)
    forecasted_congestion = db.Column(db.Text, nullable=True)

    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


    def get_congestion_list(self):
        return json.loads(self.forecasted_congestion)

with app.app_context():
    db.create_all()

def pm25_to_aqi_category(pm25):
    if 0 <= pm25 <= 12.0:
        return "Good"
    elif 12.1 <= pm25 <= 35.4:
        return "Moderate"
    elif 35.5 <= pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif 55.5 <= pm25 <= 150.4:
        return "Unhealthy"
    elif 150.5 <= pm25 <= 250.4:
        return "Very Unhealthy"
    elif pm25 >= 250.5:
        return "Hazardous"
    else:
        return "Invalid PM2.5 value"

# Params
seq_length = 10  # Past time steps for prediction
num_features = 6  # PM2.5, PM10, Noise_Level, CO, NO2, SO2

def get_latest_data():
    """Retrieve the last 'seq_length' records from the database."""
    records = TrafficData.query.order_by(TrafficData.timestamp.desc()).limit(seq_length).all()
    if len(records) < seq_length:
        return np.zeros((seq_length, num_features))  # Return empty if not enough data
    return np.array([[r.pm2_5, r.pm10, r.noise_level, r.co, r.no2, r.so2] for r in reversed(records)])

def predict_future(model, past_data, scaler, future_steps=5):
    """Future Forecast Prediction."""
    future_predictions = []
    input_seq = past_data.copy()

    for _ in range(future_steps):
        pred = model.predict(input_seq.reshape(1, seq_length, num_features))
        future_predictions.append(pred[0, 0])
        input_seq = np.roll(input_seq, -1, axis=0)
        input_seq[-1, -1] = pred[0, 0]

    future_scaled = np.zeros((future_steps, 7))
    future_scaled[:, -1] = future_predictions
    return scaler.inverse_transform(future_scaled)[:, -1]

def get_last_10_records():
    """Fetch last 10 records from the database."""
    records = TrafficData.query.order_by(TrafficData.timestamp.desc()).limit(10).all()
    return [
        {
            "timestamp": record.timestamp.isoformat(),
            "pm2_5": record.pm2_5,
            "pm10": record.pm10,
            "noise_level": record.noise_level,
            "co": record.co,
            "no2": record.no2,
            "so2": record.so2,
            "congestion": record.congestion,
            "aqi": record.aqi,
        }
        for record in reversed(records)  # Reverse for chronological order
    ]

def broadcast_last_10_records():
    """Send last 10 records to all connected WebSocket clients."""
    records = get_last_10_records()
    socketio.emit("traffic-history", {"records": records})

def generate_forecast():
    """Fetch serial data, predict congestion, save & broadcast updates."""
    global ser  # Allow reconnection to serial port if disconnected

    while True:
        time.sleep(5)
        with app.app_context():
            # Check if serial connection is active
            if ser is None or not ser.is_open:
                print("⚠️ Warning: Serial port disconnected. Attempting to reconnect...")
                try:
                    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
                    print(f"✅ Reconnected to {SERIAL_PORT} at {BAUD_RATE} baud.")
                except serial.SerialException:
                    print("❌ Failed to reconnect. Using database records instead.")
                    ser = None  # Keep it None until reconnection succeeds

            if ser and ser.in_waiting > 0:
                try:
                    data = ser.readline().decode("utf-8").strip()
                    new_entry = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    print("⚠️ Warning: Malformed serial data. Skipping...")
                    continue
            else:
                # If no serial data, fetch the latest database entry
                latest_record = TrafficData.query.order_by(TrafficData.timestamp.desc()).first()
                if latest_record:
                    print("⚠️ No serial data received. Using latest database entry.")
                    new_entry = {
                        "pm2_5": latest_record.pm2_5,
                        "pm10": latest_record.pm10,
                        "noise_level": latest_record.noise_level,
                        "co": latest_record.co,
                        "no2": latest_record.no2,
                        "so2": latest_record.so2,
                        "aqi": latest_record.aqi,
                    }
                else:
                    print("⚠️ No data in database. Waiting for new data...")
                    continue  # Skip if no data is available

            try:
                new_data = np.array([
                    new_entry["pm2_5"], new_entry["pm10"], new_entry["noise_level"],
                    new_entry["co"], new_entry["no2"], new_entry["so2"], 
                ], dtype=float)  # ✅ Ensure values retain decimal precision

                print(f"📊 New Sensor Data: {new_entry}")
            except KeyError as e:
                print(f"⚠️ Warning: Missing sensor value: {e}")
                continue

            past_data = get_latest_data()
            future_congestion = predict_future(forecast_model, past_data, forecast_scaler, future_steps=5)
            future_congestion = np.clip(future_congestion * 100, 0, 100)
            first_congestion = round(future_congestion[0], 2)

            db.session.add(TrafficData(
                pm2_5=new_entry["pm2_5"], pm10=new_entry["pm10"], noise_level=new_entry["noise_level"],
                co=new_entry["co"], no2=new_entry["no2"], so2=new_entry["so2"], aqi=pm25_to_aqi_category(new_entry["pm2_5"]),
                congestion=first_congestion
            ))
            db.session.commit()

            # 🔥 Broadcast new single entry
            socketio.emit("traffic-params", {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pm2_5": new_entry["pm2_5"],
                "pm10": new_entry["pm10"],
                "noise_level": new_entry["noise_level"],
                "co": new_entry["co"],
                "no2": new_entry["no2"],
                "so2": new_entry["so2"],
                "aqi": pm25_to_aqi_category(new_entry["pm2_5"]),
                "congestion": first_congestion
            })

            # 🔥 Broadcast last 10 records
            broadcast_last_10_records()

            # socketio.emit("traffic-forecast", {"congestion": future_congestion.tolist()})
            print(f"✅ Stored & Sent forecast: {future_congestion}")

def generate_pseudo_data():
    """Generate and insert pseudo data if there are fewer than 10 records in DailyRecord."""
    with app.app_context():
        existing_records = DailyRecord.query.count()

        if existing_records >= 10:
            print("✅ Enough records available for forecasting.")
            return

        # If fewer than 10 records, generate pseudo data
        print(f"⚠️ Only {existing_records} records. Generating pseudo data to fill up to 10 records.")
        while existing_records < 10:
            pseudo_data = DailyRecord(
                pm2_5=random.uniform(10, 60),
                pm10=random.uniform(20, 100),
                noise_level=random.uniform(40, 90),
                co=random.uniform(0.1, 2.0),
                no2=random.uniform(5, 50),
                so2=random.uniform(2, 40),
                aqi=pm25_to_aqi_category(random.uniform(10, 60)),
                forecasted_congestion=json.dumps([round(random.uniform(20, 80), 2) for _ in range(5)]),
            )
            db.session.add(pseudo_data)
            db.session.commit()

            # Increment the record count and loop again if needed
            existing_records += 1

def daily_forecast_job():
    """Run daily forecast using DailyRecord data or fallback to pseudo data."""
    generate_pseudo_data()  # Ensure at least 10 records are available

    with app.app_context():

        latest = TrafficData.query.order_by(TrafficData.timestamp.desc()).first()
        if not latest:
            print("⚠️ No data available for forecasting.")
            return

        new_record = DailyRecord(
            pm2_5=latest.pm2_5 if latest else random.uniform(10, 60),
            pm10=latest.pm10 if latest else random.uniform(20, 100),
            noise_level=latest.noise_level if latest else random.uniform(40, 90),
            co=latest.co if latest else random.uniform(0.1, 2.0),
            no2=latest.no2 if latest else random.uniform(5, 50),
            so2=latest.so2 if latest else random.uniform(2, 40),
            aqi=pm25_to_aqi_category(latest.pm2_5 if latest else 35.0),
            forecasted_congestion=json.dumps([round(random.uniform(20, 80), 2) for _ in range(5)]),
        )

        db.session.add(new_record)
        db.session.commit()

        records = DailyRecord.query.order_by(DailyRecord.timestamp.desc()).limit(10).all()

        # Proceed with forecasting
        if len(records) >= 10:
            records_array = np.array([
                [r.pm2_5, r.pm10, r.noise_level, r.co, r.no2, r.so2]
                for r in sorted(records, key=lambda x: x.timestamp)
            ])

            future_congestion = predict_future(
                model=forecast_model,
                past_data=records_array,
                scaler=forecast_scaler,
                future_steps=5
            )

            future_congestion = np.clip(future_congestion * 100, 0, 100).tolist()
            latest_record = DailyRecord.query.order_by(DailyRecord.timestamp.desc()).first()
            if latest_record:
                latest_record.forecasted_congestion = json.dumps(future_congestion)
                db.session.commit()
                print(f"✅ Updated forecasted congestion for the most recent record: {future_congestion}")
        else:
            # If still fewer than 10 after generation, create pseudo forecast
            print("⚠️ Not enough DailyRecords after generation. Using pseudo forecast.")

        print(f"✅ Daily forecast record added: {future_congestion}")


def broadcast_daily_forecast():
    """Broadcast the latest DailyRecord forecast every 5 seconds."""
    while True:
        time.sleep(5)
        with app.app_context():
            latest_daily = DailyRecord.query.order_by(DailyRecord.timestamp.desc()).first()
            if latest_daily:
                try:
                    congestion_list = json.loads(latest_daily.forecasted_congestion)
                    socketio.emit("traffic-forecast", {"congestion": congestion_list})
                    print(f"📡 Sent daily forecast: {congestion_list}")
                except json.JSONDecodeError:
                    print("⚠️ Could not decode forecasted congestion list.")


threading.Thread(target=broadcast_daily_forecast, daemon=True).start()
threading.Thread(target=generate_forecast, daemon=True).start()

scheduler = BackgroundScheduler()
scheduler.start()

scheduler.add_job(
    daily_forecast_job,
    trigger='cron',
    hour=2, minute=30 # Change time as needed
)

@app.route("/records", methods=["GET"])
def get_records():
    """API endpoint to retrieve stored records."""
    limit = request.args.get("limit", default=10, type=int)
    offset = request.args.get("offset", default=0, type=int)

    query = DailyRecord.query.order_by(DailyRecord.timestamp.desc()).offset(offset)
    
    # If limit is explicitly 0 or not provided, fetch all records
    if limit == 0:
        records = query.all()
    else:
        records = query.limit(limit).all()

    return jsonify([
        {
            "id": record.id,
            "timestamp": record.timestamp.isoformat(),
            "pm2_5": record.pm2_5,
            "pm10": record.pm10,
            "noise_level": record.noise_level,
            "co": record.co,
            "no2": record.no2,
            "so2": record.so2,
            "aqi": record.aqi,
            "congestion": record.congestion
        } for record in records
    ])

@socketio.on("connect")
def handle_connect():
    """Send last 10 records when a client connects."""
    print("✅ Client connected")
    broadcast_last_10_records()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
