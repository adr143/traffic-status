import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import pytz
import json
import random

from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, Text, DateTime

# Load CSV
df = pd.read_csv("traffic congestion analysis (1).csv")

# Combine Date and Time into timestamp
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Drop the original Date and Time if no longer needed
# df.drop(columns=['Date', 'Time'], inplace=True)

# Set up database
local_tz = pytz.timezone("Asia/Kolkata")
Base = declarative_base()

class DailyRecord(Base):
    __tablename__ = 'daily_record'
    id = Column(Integer, primary_key=True)
    pm2_5 = Column(Float, nullable=False)
    pm10 = Column(Float, nullable=False)
    noise_level = Column(Float, nullable=False)
    co = Column(Float, nullable=False)
    no2 = Column(Float, nullable=False)
    so2 = Column(Float, nullable=False)
    forecasted_congestion = Column(Text, nullable=True)
    timestamp = Column(DateTime)

# Connect to SQLite DB
engine = create_engine('sqlite:///traffic_data.db')  # Change name if needed
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Insert each row into DB
for _, row in df.iterrows():
    record = DailyRecord(
        timestamp=pd.to_datetime(row['Date'] + ' ' + row['Time'], format='%m/%d/%Y %H:%M'),
        pm2_5=row['PM2.5']/10,
        pm10=row['PM10']/10,
        noise_level=row['Noise (dB)'],
        co=round(row['CO']/20, 2),
        no2=row['NO2']/100,
        so2=row['SO2 ']/100,
        forecasted_congestion=json.dumps([round(random.uniform(.30, .67), 10) for _ in range(5)])
    )
    session.add(record)

session.commit()
session.close()
