# NeuroEmbody Data Collecting

Data collection framework for NeuroEmbody project.

## Features

- Dual RealSense camera support with automatic frame caching
- UR robot integration
- Gello handle support
- Configurable task-based data collection
- HDF5 data storage format

## Requirements

- Python 3.8+
- pyrealsense2
- OpenCV
- NumPy
- H5Py
- RealSense SDK

## Installation

```bash
git clone https://github.com/jcx0110/NeuroEmbody_data_collecting
cd NeuroEmbody_data_collecting
pip install requirements.txt
```

## Configuration

Edit `configs/hardware.yaml` to configure:
- Robot IP and port
- Camera device IDs
- Storage directory

Edit `configs/tasks.yaml` to configure:
- Task names and descriptions
- Task stages and instructions

## Usage

### Start Data Collection

```bash
bash scripts/data_collecting.sh
```

### Test Camera Connection

```bash
python data_collecting/devices/test_realsense.py --device_id "YOUR_CAMERA_SERIAL"
```

### Move Robot to Home Position

```bash
python scripts/robot_home.py
```

## Directory Structure

```
NeuroEmbody_data_collecting/
├── configs/          # Configuration files
├── data_collecting/  # Main data collection code
├── docs/             # Documentation
├── scripts/          # Utility scripts
├── tools/            # Helper tools
└── output/           # Collected data (not in git)
```

## License

[Add your license here]
