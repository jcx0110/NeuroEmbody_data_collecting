import sys
from dynamixel_sdk import *

# Configuration
DEVICENAME = '/dev/ttyUSB1'
PROTOCOL_VERSION = 2.0

# Scan IDs 1 to 10 (Robotic arms typically have 6-7 joints)
MAX_ID = 10 

# Scan these baud rates (covers factory defaults and common settings)
BAUD_LIST = [57600, 1000000, 115200, 2000000, 3000000, 4000000]

def main():
    # 1. Preparation
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        print("Failed to open port! (Check permissions or if port is busy)")
        return

    print(f"Starting full system scan (Port: {DEVICENAME})...")
    print("-" * 50)

    found_motors = {} # Record found motors {id: baud}

    # 2. Iterate through baud rates
    for baud in BAUD_LIST:
        print(f"Trying baud rate: {baud} ... ", end='', flush=True)
        portHandler.setBaudRate(baud)

        count_at_this_baud = 0
        # 3. Iterate through IDs
        for id_scan in range(1, MAX_ID + 1):
            # If this ID has already been found, skip it (avoid duplicates)
            if id_scan in found_motors:
                continue

            model, res, err = packetHandler.ping(portHandler, id_scan)
            if res == COMM_SUCCESS:
                print(f"\n   Found motor! ID: {id_scan} (Model: {model})", end='')
                found_motors[id_scan] = baud
                count_at_this_baud += 1

        if count_at_this_baud == 0:
            print("No response")
        else:
            print("") # Newline

    print("-" * 50)
    print("Scan Results Summary:")

    if not found_motors:
        print("No motors found.")
        print("   -> 1. Is the 12V power supply on?")
        print("   -> 2. Is the U2D2 side switch (TTL/485) correct?")
    else:
        sorted_ids = sorted(found_motors.keys())
        print(f"Found {len(sorted_ids)} motors in total: {sorted_ids}")
        print("Details:")
        for mid in sorted_ids:
            print(f" - ID {mid}: Baud rate {found_motors[mid]}")

        # Intelligent check
        if sorted_ids == list(range(1, len(sorted_ids) + 1)):
            print("\nPerfect! ID sequence is correct (1, 2, 3...).")
            print("   You can proceed to run GELLO directly, no need to unplug cables or change IDs!")
        else:
            print("\nIDs seem non-continuous or missing, manual adjustment might be required.")

    portHandler.closePort()

if __name__ == "__main__":
    main()