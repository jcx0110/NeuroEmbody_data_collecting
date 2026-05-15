import pickle
import threading
from typing import Any, Dict, Optional

import numpy as np
import zmq

from gello.robots.robot import Robot

DEFAULT_ROBOT_PORT = 6000
# REQ 在 recv 超时后会处于不可用状态，必须关闭并新建 socket；见 _request_rep。
DEFAULT_RECV_TIMEOUT_MS = 15000


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                # Timeout occurred - don't spam the console
                pass

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
        recv_timeout_ms: int = DEFAULT_RECV_TIMEOUT_MS,
    ):
        self._addr = f"tcp://{host}:{port}"
        self._recv_timeout_ms = int(recv_timeout_ms)
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None
        self._open_req_socket()

    def _open_req_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._recv_timeout_ms)
        self._socket.connect(self._addr)

    def _request_rep(self, method: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """REQ/REP 一轮；超时后重建 REQ（否则同一 socket 上再次 send 会永久卡住）。"""
        request: Dict[str, Any] = {"method": method}
        if args is not None:
            request["args"] = args
        payload = pickle.dumps(request)
        err_msg = (
            f"ZMQ 超时 ({self._addr}, 超时={self._recv_timeout_ms}ms)：未收到 arm_server 回复。"
            "请确认 data_collecting/core/arm_server.py 已运行、端口一致；"
            "若 arm_server 卡在 RTDE/get_observations，请检查机械臂网络。"
        )
        for attempt in range(2):
            try:
                self._socket.send(payload)
                raw = self._socket.recv()
                return pickle.loads(raw)
            except zmq.Again:
                if attempt == 0:
                    self._open_req_socket()
                    continue
                raise RuntimeError(err_msg) from None
            except zmq.ZMQError as e:
                raise RuntimeError(f"ZMQ 错误 ({self._addr}): {e}") from e
        raise RuntimeError(err_msg)

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        result = self._request_rep("num_dofs")
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        result = self._request_rep("get_joint_state")
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        result = self._request_rep(
            "command_joint_state", args={"joint_state": joint_state}
        )
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        result = self._request_rep("get_observations")
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        self._context.term()
