import yaml
import sys
from pathlib import Path

try:
    from data_collecting.utils.logger import Logger as log
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data_collecting.utils.logger import Logger as log


class ConfigLoader:
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent
        self.config_dir = project_root / "configs"
        
        self.hardware = self._load_yaml("hardware.yaml")
        self.tasks = self._load_yaml("tasks.yaml")

    def _load_yaml(self, file_name):
        path = self.config_dir / file_name
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.error(f"Failed to load {file_name}: {e}")
            return {}

    def get_robot(self):
        return self.hardware.get('robot', {})

    def get_camera(self):
        return self.hardware.get('camera', {})

    def get_gripper(self):
        return self.hardware.get('gripper', {})

    def get_storage_dir(self):
        return self.hardware.get('storage', {}).get('save_root', './output')

    def get_task(self):
        active_name = self.tasks.get("task")
        if not active_name:
            log.error("No default 'task' specified in tasks.yaml")
            return {}
        
        task_info = self.tasks.get(active_name, {})
        if not task_info:
            log.error(f"Task '{active_name}' details not found in tasks.yaml")
            return {}
            
        task_info['name'] = active_name 
        return task_info


if __name__ == "__main__":
    print("\n--- ConfigLoader Check ---\n")
    cfg = ConfigLoader()

    print(f"1. Robot:   {cfg.get_robot()}")
    print(f"2. Camera:  {cfg.get_camera()}")
    print(f"3. Gripper: {cfg.get_gripper()}")
    print(f"4. Storage: {cfg.get_storage_dir()}")
    
    print("\n5. Current Task Details:")
    task = cfg.get_task()
    print(f"   Name: {task.get('name')}")
    print(f"   Stage_num: {task.get('stages_num')}")
    print(f"   Desc: {task.get('description')}")
    print(f"   Steps: {task.get('instructions')}")
    
    print("\n--- Check Complete ---")