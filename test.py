import serial
import json

# Serial Port Configuration (Adjust if needed)
SERIAL_PORT = "COM9"  # Change to the correct port (e.g., "/dev/ttyAMA0" for GPIO serial)
BAUD_RATE = 115200  # Match with Arduino's baud rate

# Open Serial Connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"Listening on {SERIAL_PORT} at {BAUD_RATE} baud...")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

def read_serial():
    """Reads and parses incoming JSON data from the serial port."""
    try:
        while True:
            if ser.in_waiting > 0:  # Check if data is available
                data = ser.readline().decode("utf-8").strip()
                
                if data:
                    try:
                        # Parse JSON data
                        sensor_data = json.loads(data)
                        
                        # Print received data
                        print("\nReceived Sensor Data:")
                        print(f"PM1.0: {sensor_data['pm1_0']} µg/m³")
                        print(f"PM2.5: {sensor_data['pm2_5']} µg/m³")
                        print(f"PM10: {sensor_data['pm10']} µg/m³")
                        print(f"Noise Level: {sensor_data['noise_level']}")
                        print(f"CO2: {sensor_data['co2']} (Raw ADC)")
                        print(f"NO2: {sensor_data['no2']} (Raw ADC)")
                        print(f"SO2: {sensor_data['so2']} (Raw ADC)")
                    
                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {data}")
    
    except KeyboardInterrupt:
        print("\nSerial reading stopped.")
        ser.close()

if __name__ == "__main__":
    read_serial()
