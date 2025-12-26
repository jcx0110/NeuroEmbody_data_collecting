# RealSense Camera Setup Guide

## Table of Contents
1. [Get Camera Device ID](#get-camera-device-id)
2. [Methods to Set Camera ID](#methods-to-set-camera-id)
3. [FAQ](#faq)

---

## Get Camera Device ID

### Method 1: Use Tool Script (Recommended)

Run the following command to list all connected RealSense cameras:

```bash
cd NeuroEmbody_data_collecting
python tools/list_realsense_cameras.py
```

Example output:
```
Detected 2 RealSense devices:

================================================================================

Device #1:
  Name: Intel RealSense D435
  Serial Number: 123456789012
  Firmware Version: 05.13.00.50

  Tip: To use this device, set device_id = '123456789012'
--------------------------------------------------------------------------------

Device #2:
  Name: Intel RealSense D435i
  Serial Number: 987654321098
  Firmware Version: 05.13.00.50

  Tip: To use this device, set device_id = '987654321098'
--------------------------------------------------------------------------------
```

### Method 2: Use rs-enumerate-devices Command

If RealSense SDK is installed, you can use:

```bash
rs-enumerate-devices
```

---

## Methods to Set Camera ID

### Method 1: Via Configuration File (Recommended, Permanent Setting)

Edit the `configs/hardware.yaml` file:

```yaml
cameras:
  front:
    type: "realsense"
    device_id: "123456789012"  # Serial number of front camera
    width: 640
    height: 480
    fps: 30
  side:
    type: "realsense"
    device_id: "987654321098"  # Serial number of side camera
    width: 640
    height: 480
    fps: 30
```

**Note**: Replace `device_id` with the actual serial number obtained from the tool script.

### Method 2: Via Command Line Arguments (Temporary Setting, Overrides Config File)

```bash
python data_collecting/core/run_data_collecting.py \
  --front_camera_id "123456789012" \
  --side_camera_id "987654321098"
```

### Method 3: Use "default" (Auto-select)

If `device_id` is not specified or set to `"default"`, the system will automatically select the first available camera.

**Warning**: If there are multiple cameras, using `"default"` may result in different cameras being selected on each run. It is recommended to explicitly specify the serial number.

---

## Configuration Examples

### Example 1: Two Different Cameras

Assuming you have two RealSense D435 cameras:

```yaml
cameras:
  front:
    type: "realsense"
    device_id: "123456789012"  # Front camera serial number
    width: 640
    height: 480
    fps: 30
  side:
    type: "realsense"
    device_id: "987654321098"  # Side camera serial number
    width: 640
    height: 480
    fps: 30
```

### Example 2: Using the Same Camera (Not Recommended)

If you only have one camera, you can set the same serial number, but this will cause conflicts:

```yaml
cameras:
  front:
    type: "realsense"
    device_id: "123456789012"
  side:
    type: "realsense"
    device_id: "123456789012"  # Same serial number will cause errors
```

**Recommendation**: If you only have one camera, use only `front_camera_id` and set `side_camera_id` to `"default"` or leave it empty.

---

## FAQ

### Q1: How to confirm the camera is properly connected?

Run the tool script:
```bash
python tools/list_realsense_cameras.py
```

If no devices are found, check:
- Whether the USB cable supports data transfer (not charging-only)
- Whether the USB port is working properly
- Whether RealSense SDK is installed

### Q2: Getting "Failed to start RealSense" error?

Possible causes:
1. **Incorrect device ID**: Check if the serial number is correct
2. **Device already in use**: Close other programs using the camera
3. **Insufficient USB bandwidth**: Try using a USB 3.0 port
4. **Permission issues**: On Linux, you may need to add the user to the `video` group:
   ```bash
   sudo usermod -a -G video $USER
   ```

### Q3: How to test a single camera?

You can modify the code to temporarily use only one camera:

```python
# Use only front camera
front_cam = RealSenseCamera(device_id="123456789012", width=640, height=480, fps=30)
side_cam = None  # Or comment out side camera related code
```

### Q4: What is the format of camera ID?

The `device_id` for RealSense cameras should be the **Serial Number**, typically a 12-digit numeric string, for example: `"123456789012"`.

### Q5: Will the camera ID change?

**Generally, NO.** The camera serial number (device_id) is a **permanent hardware identifier** that is:
- **Fixed**: Burned into the camera's firmware at manufacturing time
- **Unique**: Each camera has a unique serial number
- **Persistent**: Remains the same regardless of:
  - USB port connection
  - Computer restart
  - Software reinstallation
  - Operating system changes

**However**, the serial number may appear to "change" in these rare cases:
1. **Different camera**: If you physically replace the camera with a different unit
2. **Firmware corruption**: Extremely rare, only if firmware is severely corrupted
3. **Hardware failure**: If the camera's internal memory storing the serial number fails

**Best Practice**: Once you identify your camera's serial number, you can safely use it in your configuration file. It will remain stable for the lifetime of the camera.

### Q6: Can I switch cameras at runtime?

No. Camera ID must be set when the program starts. If you need to switch, you need to restart the program.

---

## Quick Checklist

- [ ] Run `python tools/list_realsense_cameras.py` to get device serial numbers
- [ ] Set the correct `device_id` in `configs/hardware.yaml`
- [ ] Confirm that the two cameras use different serial numbers
- [ ] Test if cameras can start normally
- [ ] Confirm that images can be displayed properly

---

## Related Files

- Configuration file: `configs/hardware.yaml`
- Tool script: `tools/list_realsense_cameras.py`
- Camera class: `data_collecting/devices/realsense.py`
- Main program: `data_collecting/core/run_data_collecting.py`
