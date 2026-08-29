import socket
import threading
import time
from collections import OrderedDict
from enum import Enum


class RobotiqGripper:
    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"

    STA = "STA"
    PRE = "PRE"
    OBJ = "OBJ"

    ENCODING = "utf-8"

    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _set_vars(self, values: OrderedDict) -> bool:
        cmd = "SET"
        for variable, value in values.items():
            cmd += f" {variable} {value}"
        cmd += "\n"
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            reply = self.socket.recv(1024)
        return reply == b"ack"

    def _set_var(self, variable: str, value: int) -> bool:
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str) -> int:
        with self.command_lock:
            self.socket.sendall(f"GET {variable}\n".encode(self.ENCODING))
            reply = self.socket.recv(1024)
        name, value = reply.decode(self.ENCODING).split()
        if name != variable:
            raise ValueError(f"Unexpected gripper response: {reply!r}")
        return int(value)

    def _reset(self) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)

    def is_active(self) -> bool:
        return self.GripperStatus(self._get_var(self.STA)) == self.GripperStatus.ACTIVE

    def activate(self, auto_calibrate: bool = True) -> None:
        if not self.is_active():
            self._reset()
            while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
                time.sleep(0.01)
            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while self._get_var(self.ACT) != 1 or self._get_var(self.STA) != 3:
                time.sleep(0.01)
        if auto_calibrate:
            self.auto_calibrate()

    def get_current_position(self) -> int:
        return self._get_var(self.POS)

    def move(self, position: int, speed: int, force: int):
        position = int(np_clip(position, self._min_position, self._max_position))
        speed = int(np_clip(speed, self._min_speed, self._max_speed))
        force = int(np_clip(force, self._min_force, self._max_force))
        values = OrderedDict(
            [(self.POS, position), (self.SPE, speed), (self.FOR, force), (self.GTO, 1)]
        )
        return self._set_vars(values), position

    def move_and_wait_for_pos(self, position: int, speed: int, force: int):
        ok, target = self.move(position, speed, force)
        if not ok:
            raise RuntimeError("Failed to send gripper move command.")
        while self._get_var(self.PRE) != target:
            time.sleep(0.001)
        obj = self._get_var(self.OBJ)
        while self.ObjectStatus(obj) == self.ObjectStatus.MOVING:
            obj = self._get_var(self.OBJ)
        return self._get_var(self.POS), self.ObjectStatus(obj)

    def auto_calibrate(self) -> None:
        pos, status = self.move_and_wait_for_pos(self._min_position, 64, 1)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Failed to open gripper during calibration: {status}")
        self._min_position = pos

        pos, status = self.move_and_wait_for_pos(self._max_position, 64, 1)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Failed to close gripper during calibration: {status}")
        self._max_position = pos

        pos, status = self.move_and_wait_for_pos(self._min_position, 64, 1)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Failed to reopen gripper during calibration: {status}")
        self._min_position = pos


def np_clip(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(int(value), max_value))
