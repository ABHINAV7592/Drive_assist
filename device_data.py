import serial

def read_from_arduino(port="COM8", baud_rate=9600):
    try:
        # Open the serial port
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to {port} at {baud_rate} baud")
        
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()
                print("Received:", data)
    
    except serial.SerialException as e:
        print("Error:", e)
    except KeyboardInterrupt:
        print("\nStopping communication.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    read_from_arduino("COM8")  # Change COM3 to your Arduino's port
