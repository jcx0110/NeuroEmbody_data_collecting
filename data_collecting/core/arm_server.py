from dataclasses import dataclass
from pathlib import Path
import sys
import tyro

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from data_collecting.utils.logger import Logger as log
    from data_collecting.utils.config_loader import ConfigLoader
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from data_collecting.utils.logger import Logger as log
    from data_collecting.utils.config_loader import ConfigLoader

from third_party.gello_software.gello.robots.robot import BimanualRobot, PrintRobot
from third_party.gello_software.gello.zmq_core.robot_node import ZMQServerRobot

cfg_loader = ConfigLoader()
robot_cfg = cfg_loader.get_robot()

@dataclass
class Args:
    robot: str = robot_cfg.get("type", "ur") 
    robot_port: int = robot_cfg.get("port", 6001)
    hostname: str = "127.0.0.1"
    robot_ip: str = robot_cfg.get("ip", "192.168.1.10")


def launch_robot_server(args: Args):
    port = args.robot_port
    log.info(f"Launching Robot Server: {args.robot} on port {port}")

    if args.robot.startswith("sim_"):
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        from third_party.gello_software.gello.robots.sim_robot import MujocoRobotServer

        if args.robot == "sim_ur":
            xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
            gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
            server = MujocoRobotServer(xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname)
        
        elif args.robot == "sim_yam":
            xml = MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"
            server = MujocoRobotServer(xml_path=xml, gripper_xml_path=None, port=port, host=args.hostname)
        
        elif args.robot == "sim_panda":
            xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
            server = MujocoRobotServer(xml_path=xml, gripper_xml_path=None, port=port, host=args.hostname)
        
        elif args.robot == "sim_xarm":
            xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
            server = MujocoRobotServer(xml_path=xml, gripper_xml_path=None, port=port, host=args.hostname)
        
        server.serve()

    else:
        if args.robot == "xarm":
            from third_party.gello_software.gello.robots.xarm_robot import XArmRobot
            log.info(f"Initializing xArm at {args.robot_ip}...")
            robot = XArmRobot(ip=args.robot_ip)

        elif args.robot == "ur":
            from data_collecting.robot.ur import URRobot
            log.info(f"Initializing UR Robot at {args.robot_ip}...")
            robot = URRobot(robot_ip=args.robot_ip)

        elif args.robot == "panda":
            from third_party.gello_software.gello.robots.panda import PandaRobot
            robot = PandaRobot(robot_ip=args.robot_ip)

        elif args.robot == "bimanual_ur":
            from third_party.gello_software.gello.robots.ur import URRobot
            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)

        elif args.robot == "yam":
            from third_party.gello_software.gello.robots.yam import YAMRobot
            robot = YAMRobot(channel="can0")

        elif args.robot in ["none", "print"]:
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )

        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        log.success(f"ZMQ Server running at {args.hostname}:{port}")
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))