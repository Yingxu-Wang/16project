"""
智能仓储分拣系统 v2.0
功能特性：
1. 多线程并行处理架构
2. 三级容错机制
3. 动态参数热更新
4. 硬件状态监控
5. 模拟器支持
6. 自动化校准系统
7. 数据可视化界面
"""

# 标准库导入
import sys
import os
import cv2
import time
import json
import serial
import logging
import argparse
import subprocess
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from threading import Thread, Event, Lock, Semaphore
from queue import Queue, LifoQueue, PriorityQueue
from voice_assistant import VoiceAssistant, DeliveryAnnouncer

# 第三方库导入
import psutil
from robomaster import robot, camera
from ultralytics import YOLO
from serial.tools import list_ports

# ---------------------------- 系统常量 ----------------------------
VERSION = "2.1.0"
DEFAULT_CONFIG_PATH = "config/system_config.json"
CALIBRATION_DATA_PATH = "config/calibration_data.bin"

# ---------------------------- 自定义异常 ----------------------------
class HardwareException(Exception):
    """硬件通信异常基类"""
    pass

class CameraException(HardwareException):
    """摄像头异常"""
    pass

class ArmException(HardwareException):
    """机械臂异常"""
    pass

class ArduinoException(HardwareException):
    """Arduino通信异常"""
    pass

# ---------------------------- 配置结构体 ----------------------------
@dataclass
class VisionConfig:
    model_path: str = "models/yolov8n-seg.pt"
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.65
    fps_limit: int = 30
    exposure: float = -5.0
    white_balance: str = "auto"

@dataclass
class ArmConfig:
    home_position: Tuple[int, int] = (0, 0)
    extended_position: Tuple[int, int] = (150, -30)
    speed_profile: Dict[str, float] = None
    max_retries: int = 3
    collision_threshold: int = 500

@dataclass
class ChassisConfig:
    base_speed: float = 0.3
    precision_mode_speed: float = 0.1
    rotation_speed: float = 30.0  # deg/s
    obstacle_distance: float = 0.5  # meters
    emergency_stop_delay: float = 0.2  # seconds

@dataclass
class IoConfig:
    arduino_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200
    timeout: float = 2.0
    max_retries: int = 5
    heartbeat_interval: float = 1.0

@dataclass
class SystemConfig:
    vision: VisionConfig
    arm: ArmConfig
    chassis: ChassisConfig
    io: IoConfig
    log_level: str = "INFO"
    simulation_mode: bool = False
    enable_debug_ui: bool = True

# ---------------------------- 硬件抽象层 ----------------------------
class HardwareManager:
    """硬件资源统一管理类"""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._init_resources()
            return cls._instance

    def _init_resources(self):
        """初始化硬件资源"""
        self.robot = None
        self.camera = None
        self.serial_ports = {}
        self.hardware_status = {
            'camera': False,
            'arm': False,
            'chassis': False,
            'gripper': False,
            'arduino': False
        }
        self.resource_locks = {
            'camera': Semaphore(1),
            'arm': Semaphore(1),
            'chassis': Semaphore(1),
            'serial': Semaphore(1)
        }

    def initialize_robot(self, conn_type="ap", max_retries=3):
        """初始化机器人平台"""
        for attempt in range(max_retries):
            try:
                self.robot = robot.Robot()
                self.robot.initialize(conn_type=conn_type)
                self.camera = self.robot.camera
                self.hardware_status.update({
                    'camera': True,
                    'arm': True,
                    'chassis': True,
                    'gripper': True
                })
                logging.info("机器人平台初始化成功")
                return True
            except Exception as e:
                logging.error(f"机器人初始化失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(1)
        raise HardwareException("机器人平台初始化失败")

    def connect_arduino(self, config: IoConfig):
        """连接Arduino设备"""
        with self.resource_locks['serial']:
            if self.hardware_status['arduino']:
                return True

            for port in list_ports.comports():
                if config.arduino_port in port.device:
                    try:
                        self.serial_ports['arduino'] = serial.Serial(
                            port=port.device,
                            baudrate=config.baud_rate,
                            timeout=config.timeout
                        )
                        time.sleep(2)  # 等待握手
                        self._arduino_handshake()
                        self.hardware_status['arduino'] = True
                        logging.info(f"Arduino连接成功: {port.device}")
                        return True
                    except Exception as e:
                        logging.error(f"Arduino连接失败: {str(e)}")

            logging.error("未找到可用的Arduino设备")
            return False

    def _arduino_handshake(self):
        """Arduino握手协议"""
        self.serial_ports['arduino'].write(b'HELLO\n')
        response = self.serial_ports['arduino'].readline().decode().strip()
        if response != "READY":
            raise ArduinoException("Arduino握手失败")

    def get_system_metrics(self):
        """获取系统性能指标"""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'temperature': self._read_cpu_temp(),
            'disk_usage': psutil.disk_usage('/').percent
        }

    def _read_cpu_temp(self):
        """读取CPU温度（仅限Jetson）"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return 0.0

# ---------------------------- 视觉处理管道 ----------------------------
class VisionPipeline:
    """增强型视觉处理管道"""
    def __init__(self, config: VisionConfig):
        self.config = config
        self.model = YOLO(self.config.model_path)
        self.fps_counter = FPSCounter()
        self.calibration = CalibrationManager()
        self.frame_buffer = LifoQueue(maxsize=3)
        self.processing_lock = Lock()
        self._init_pipeline()

    def _init_pipeline(self):
        """初始化视觉管道"""
        self.class_names = self.model.names
        self._warmup_model()
        self.calibration.load(CALIBRATION_DATA_PATH)
        logging.info("视觉管道初始化完成")

    def _warmup_model(self):
        """模型预热"""
        dummy_input = np.random.randn(*self.config.input_size, 3).astype(np.uint8)
        for _ in range(3):
            self.model(dummy_input, verbose=False)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理单个帧"""
        with self.processing_lock:
            self.fps_counter.update()

            # 预处理
            calibrated_frame = self.calibration.apply(frame)
            resized_frame = cv2.resize(calibrated_frame, self.config.input_size)

            # 推理
            results = self.model(
                resized_frame,
                imgsz=self.config.input_size,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False
            )[0]

            # 后处理
            processed = {
                'timestamp': time.time(),
                'original_frame': frame,
                'processed_frame': resized_frame,
                'detections': [],
                'performance': {
                    'fps': self.fps_counter.fps,
                    'inference_time': results.speed['inference'],
                    'preprocess_time': results.speed['preprocess'],
                    'postprocess_time': results.speed['postprocess']
                }
            }

            for box, conf, cls, track_id in zip(results.boxes.xyxy,
                                              results.boxes.conf,
                                              results.boxes.cls,
                                              results.boxes.id):
                detection = {
                    'bbox': box.cpu().numpy(),
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.class_names[int(cls)],
                    'track_id': int(track_id) if track_id else None
                }
                processed['detections'].append(detection)

            self.frame_buffer.put(processed)
            return processed

    def get_latest_frame(self):
        """获取最新处理帧"""
        return self.frame_buffer.get() if not self.frame_buffer.empty() else None

class CalibrationManager:
    """相机校准管理器"""
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.undistort_map = None

    def load(self, file_path: str):
        """加载校准数据"""
        try:
            with np.load(file_path) as data:
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                self._init_undistort_map()
            logging.info("相机校准数据加载成功")
        except Exception as e:
            logging.warning(f"无法加载校准数据: {str(e)}")

    def _init_undistort_map(self):
        """初始化去畸变映射"""
        h, w = 1080, 1920  # 根据实际分辨率调整
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        self.undistort_map = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_camera_matrix, (w,h), cv2.CV_16SC2)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """应用校准"""
        if self.undistort_map is not None:
            return cv2.remap(frame, *self.undistort_map, cv2.INTER_LINEAR)
        return frame

class FPSCounter:
    """高性能FPS计数器"""
    def __init__(self, window_size=30):
        self.times = []
        self.window_size = window_size
        self.fps = 0.0

    def update(self):
        """更新计数器"""
        now = time.time()
        self.times.append(now)
        if len(self.times) > self.window_size:
            self.times.pop(0)
        if len(self.times) >= 2:
            self.fps = (len(self.times)-1) / (self.times[-1]-self.times[0])

# ---------------------------- 运动控制子系统 ----------------------------
class MotionController:
    """增强型运动控制器"""
    def __init__(self, hardware: HardwareManager, config: SystemConfig):
        self.hw = hardware
        self.config = config
        self.current_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.target_pose = None
        self.trajectory_queue = PriorityQueue()
        self.emergency_stop = Event()
        self._init_kinematic_model()

    def _init_kinematic_model(self):
        """初始化运动学模型"""
        # 简化的差速驱动模型参数
        self.wheel_radius = 0.0762  # 米
        self.wheel_base = 0.3302    # 米
        self.max_linear_speed = 1.0 # 米/秒
        self.max_angular_speed = 60.0 # 度/秒

    def move_to(self, x: float, y: float, theta: float = None):
        """移动到指定位姿"""
        self.target_pose = (x, y, theta)
        self._plan_trajectory()
        self._execute_trajectory()

    def _plan_trajectory(self):
        """轨迹规划（简化版）"""
        # TODO: 实现完整的路径规划算法
        dx = self.target_pose[0] - self.current_pose['x']
        dy = self.target_pose[1] - self.current_pose['y']
        distance = np.hypot(dx, dy)
        angle = np.degrees(np.arctan2(dy, dx))

        self.trajectory_queue.put((
            1,  # 优先级
            {'type': 'rotate', 'angle': angle}
        ))
        self.trajectory_queue.put((
            2,  # 优先级
            {'type': 'move', 'distance': distance}
        ))
        if self.target_pose[2] is not None:
            self.trajectory_queue.put((
                3,  # 优先级
                {'type': 'rotate', 'angle': self.target_pose[2]}
            ))

    def _execute_trajectory(self):
        """执行轨迹"""
        while not self.trajectory_queue.empty():
            if self.emergency_stop.is_set():
                logging.warning("紧急停止触发")
                return

            _, action = self.trajectory_queue.get()
            if action['type'] == 'rotate':
                self._rotate(action['angle'])
            elif action['type'] == 'move':
                self._linear_move(action['distance'])

    def _rotate(self, degrees: float):
        """精确旋转控制"""
        speed = np.clip(abs(degrees)/180.0, 0.1, 1.0) * self.config.chassis.rotation_speed
        direction = np.sign(degrees)

        with self.hw.resource_locks['chassis']:
            self.hw.robot.chassis.move(
                z=direction*abs(degrees),
                z_speed=speed
            ).wait_for_completed()
            self.current_pose['z'] += degrees

    def _linear_move(self, distance: float):
        """直线运动控制"""
        speed = self.config.chassis.base_speed
        time_needed = abs(distance) / speed

        with self.hw.resource_locks['chassis']:
            self.hw.robot.chassis.move(
                x=distance,
                xy_speed=speed
            ).wait_for_completed(timeout=time_needed+1.0)
            self.current_pose['x'] += distance * np.cos(np.radians(self.current_pose['z']))
            self.current_pose['y'] += distance * np.sin(np.radians(self.current_pose['z']))

    def emergency_stop_handler(self):
        """紧急停止处理"""
        self.emergency_stop.set()
        with self.hw.resource_locks['chassis']:
            self.hw.robot.chassis.drive_speed(x=0, y=0, z=0)
        logging.warning("紧急停止已执行")

# ---------------------------- 主控制系统 ----------------------------
class WarehouseAutomationSystem:
    """智能仓储主控制系统"""
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config = self._load_config(config_path)
        self._init_logging()
        self.hw_manager = HardwareManager()
        self.vision_pipe = VisionPipeline(self.config.vision)
        self.motion_ctrl = MotionController(self.hw_manager, self.config)
        self.state_machine = StateMachine()
        self.ui = DebugUI(enabled=self.config.enable_debug_ui)
        self._running = Event()
        self._init_complete = False
        self.voice = VoiceAssistant()
        self.announcer = DeliveryAnnouncer(self.voice)

    def _load_config(self, path: str) -> SystemConfig:
        """加载系统配置"""
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
            return SystemConfig(
                vision=VisionConfig(**config_data['vision']),
                arm=ArmConfig(**config_data['arm']),
                chassis=ChassisConfig(**config_data['chassis']),
                io=IoConfig(**config_data['io']),
                log_level=config_data.get('log_level', 'INFO'),
                simulation_mode=config_data.get('simulation_mode', False),
                enable_debug_ui=config_data.get('enable_debug_ui', True)
            )
        except Exception as e:
            logging.error(f"配置加载失败: {str(e)}")
            return SystemConfig(
                vision=VisionConfig(),
                arm=ArmConfig(),
                chassis=ChassisConfig(),
                io=IoConfig()
            )

    def _init_logging(self):
        """初始化日志系统"""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('system.log'),
                logging.StreamHandler()
            ]
        )
        logging.info(f"系统初始化完成，版本: {VERSION}")

    def startup_sequence(self):
        """系统启动序列"""
        try:
            # 硬件初始化
            if not self.config.simulation_mode:
                self.hw_manager.initialize_robot()
                self.hw_manager.connect_arduino(self.config.io)

            # 启动子系统
            self.vision_pipe.start()
            self.motion_ctrl.calibrate()
            self.state_machine.transition_to(State.READY)
            self._init_complete = True
            logging.info("系统启动完成")

        except HardwareException as e:
            logging.error(f"硬件初始化失败: {str(e)}")
            self.shutdown()
	  self.voice.start()
          self.announcer.start()


    def main_loop(self):
       try:
            while self._running.is_set():
        """主控制循环"""
        self._running.set()
        last_update = time.time()

        try:
            while self._running.is_set():
                current_time = time.time()
                delta_time = current_time - last_update
                last_update = current_time

                # 状态机更新
                self.state_machine.update(delta_time)

                # 处理视觉数据
                frame_data = self.vision_pipe.get_latest_frame()
                if frame_data:
                    self._process_vision_data(frame_data)

                # 更新UI
                self.ui.update(
                    system_status=self.get_system_status(),
                    frame_data=frame_data
                )

                time.sleep(0.01)  # 控制循环频率

        except KeyboardInterrupt:
            logging.info("用户中断请求")
        finally:
            self.shutdown()
		# 处理语音命令
                self._process_voice_commands()
                
                # 运动时自动播报
                if self.state_machine.current_state in [State.TRACKING, State.GRASPING]:
                    self.announcer.run()
	def _process_voice_commands(self):
              if not self.voice.command_queue.empty():
                   command = self.voice.command_queue.get()
                if command == "start_delivery":
                  self._start_delivery_procedure()
               elif command == "emergency_stop":
                   self.motion_ctrl.emergency_stop_handler()	

    def _process_vision_data(self, data: Dict):
        """处理视觉数据"""
        # TODO: 实现具体的物体识别和定位逻辑
        pass

    def get_system_status(self) -> Dict:
        """获取系统状态报告"""
        metrics = self.hw_manager.get_system_metrics()
        return {
            'state': self.state_machine.current_state.name,
            'hardware': self.hw_manager.hardware_status,
            'performance': metrics,
            'vision': {
                'fps': self.vision_pipe.fps_counter.fps,
                'object_count': len(self.vision_pipe.get_latest_frame()['detections'])
                if self.vision_pipe.get_latest_frame() else 0
            }
        }

    def shutdown(self):
        """系统关闭流程"""
        self._running.clear()
        self.vision_pipe.stop()
        self.hw_manager.shutdown()
        self.ui.close()
        logging.info("系统已安全关闭")

	 def main_loop(self):
        """增强型主控制循环"""
        self._running.set()
        last_vision_update = time.time()
        operation_timeout = 30.0  # 单次操作超时时间

        try:
            while self._running.is_set():
                current_time = time.time()
                delta_time = current_time - last_vision_update
                last_vision_update = current_time

                # 状态机更新
                self.state_machine.update(delta_time)

                # 获取最新视觉数据
                frame_data = self.vision_pipe.get_latest_frame()
                if frame_data:
                    self._process_vision_data(frame_data)

                # 执行状态相关操作
                self._execute_state_operations()

                # 系统健康检查
                if not self._health_check():
                    self.state_machine.transition_to(State.ERROR)

                # 更新UI
                self.ui.update(
                    system_status=self.get_system_status(),
                    frame_data=frame_data
                )

                time.sleep(0.01)

        except KeyboardInterrupt:
            logging.info("用户中断请求")
        finally:
            self.shutdown()
	self.voice.stop()
        self.announcer.stop()

    def _execute_state_operations(self):
        """状态相关操作执行器"""
        state = self.state_machine.current_state

        if state == State.READY:
            self._ready_state_operations()
        elif state == State.DETECTING:
            self._detecting_state_operations()
        elif state == State.LIFTING:
            self._lifting_state_operations()
        elif state == State.PUSHING:
            self._pushing_state_operations()
        elif state == State.RETURNING:
            self._returning_state_operations()
        elif state == State.ERROR:
            self._error_state_operations()

    def _ready_state_operations(self):
        """就绪状态操作"""
        # 检查是否有待处理订单
        if self.order_queue.has_order():
            self.state_machine.transition_to(State.DETECTING)

    def _detecting_state_operations(self):
        """检测状态操作"""
        # 持续获取最新帧数据
        frame_data = self.vision_pipe.get_latest_frame()
        if not frame_data:
            return

        # 寻找可抓取目标
        target = self._find_valid_target(frame_data['detections'])
        if target:
            self.current_target = target
            self.state_machine.transition_to(State.TRACKING)

    def _lifting_state_operations(self):
        """升降平台控制流程"""
        if not hasattr(self, 'lift_operation_started'):
            # 初始化升降操作
            self.lift_operation_started = time.time()
            target_layer = self._calculate_target_layer(self.current_target)
            
            if not self.motion_ctrl.operate_lift(target_layer):
                self.state_machine.transition_to(State.ERROR)
                return
            
            # 启动升降超时监控
            self.lift_timeout = time.time() + self.motion_ctrl.LAYER_HEIGHTS[target_layer] + 5.0

        # 检查超时
        if time.time() > self.lift_timeout:
            logging.error("升降操作超时")
            self.state_machine.transition_to(State.ERROR)
            return

        # 检查Arduino状态
        if self._check_lift_complete():
            self.state_machine.transition_to(State.PUSHING)
            del self.lift_operation_started

    def _pushing_state_operations(self):
        """推送操作控制"""
        if not hasattr(self, 'push_operation_started'):
            # 发送推送命令
            if self.hw_manager.send_arduino_command("PUSH"):
                self.push_operation_started = time.time()
                self.push_timeout = time.time() + 5.0  # 推送超时时间
            else:
                self.state_machine.transition_to(State.ERROR)
                return

        # 检查推送完成
        if self._check_push_complete():
            self.state_machine.transition_to(State.RETURNING)
            del self.push_operation_started
        elif time.time() > self.push_timeout:
            logging.error("推送操作超时")
            self.state_machine.transition_to(State.ERROR)

    def _returning_state_operations(self):
        """返回原点操作"""
        if not hasattr(self, 'return_operation_started'):
            # 发送返回命令
            if self.hw_manager.send_arduino_command("RETURN"):
                self.return_operation_started = time.time()
                self.return_timeout = time.time() + 10.0  # 返回超时时间
            else:
                self.state_machine.transition_to(State.ERROR)
                return

        # 检查返回完成
        if self._check_return_complete():
            self.state_machine.transition_to(State.READY)
            del self.return_operation_started
            self._complete_order()
        elif time.time() > self.return_timeout:
            logging.error("返回操作超时")
            self.state_machine.transition_to(State.ERROR)

    def _check_lift_complete(self) -> bool:
        """检查升降是否完成"""
        # 通过串口查询状态
        return self.hw_manager.send_arduino_command("STATUS", timeout=1.0)

    def _check_push_complete(self) -> bool:
        """检查推送是否完成"""
        # 通过限位开关或编码器反馈判断
        return self.hw_manager.send_arduino_command("CHECK_PUSH", timeout=1.0)

    def _check_return_complete(self) -> bool:
        """检查是否返回原点"""
        return self.hw_manager.send_arduino_command("CHECK_HOME", timeout=1.0)

    def _health_check(self) -> bool:
        """系统健康检查"""
        metrics = self.hw_manager.get_system_metrics()
        return (
            metrics['cpu_usage'] < 90 and
            metrics['memory_usage'] < 85 and
            metrics['temperature'] < 85 and
            not self.emergency_stop.is_set()
        )

    def _complete_order(self):
        """完成订单处理"""
        # 更新订单状态
        self.order_queue.complete_current_order()
        # 重置当前目标
        self.current_target = None
        logging.info("订单处理完成")

    def _calculate_target_layer(self, detection) -> int:
        """计算目标货架层数"""
        # 基于检测框位置判断
        img_center_y = detection['bbox'][1] + (detection['bbox'][3] - detection['bbox'][1])/2
        frame_height = self.config.vision.input_size[1]
        
        if img_center_y < frame_height/3:
            return 3
        elif img_center_y < frame_height*2/3:
            return 2
        else:
            return 1

    def _error_state_operations(self):
        """错误状态处理"""
        # 紧急停止所有设备
        self.motion_ctrl.emergency_stop_handler()
        # 尝试恢复
        if self._try_recover():
            self.state_machine.transition_to(State.READY)
        else:
            self.shutdown()

    def _try_recover(self) -> bool:
        """尝试系统恢复"""
        recovery_attempts = 3
        for attempt in range(recovery_attempts):
            logging.info(f"尝试系统恢复 ({attempt+1}/{recovery_attempts})")
            
            # 重置硬件
            if self.hw_manager.reset_hardware():
                # 重新初始化视觉
                self.vision_pipe.restart()
                # 清除错误状态
                self.emergency_stop.clear()
                return True
            
            time.sleep(1)
        
        return False
# ---------------------------- 状态机实现 ----------------------------
class State(Enum):
    INIT = auto()
    CALIBRATING = auto()
    READY = auto()
    DETECTING = auto()
    TRACKING = auto()
    GRASPING = auto()
    LIFTING = auto()
    ERROR = auto()

class StateMachine:
    """增强型状态机"""
    def __init__(self):
        self.current_state = State.INIT
        self.state_handlers = {
            State.INIT: self._handle_init,
            State.CALIBRATING: self._handle_calibrating,
            State.READY: self._handle_ready,
            State.DETECTING: self._handle_detecting,
            State.TRACKING: self._handle_tracking,
            State.GRASPING: self._handle_grasping,
            State.LIFTING: self._handle_lifting,
            State.ERROR: self._handle_error
        }
        self.transitions = {
            State.INIT: [State.CALIBRATING],
            State.CALIBRATING: [State.READY, State.ERROR],
            State.READY: [State.DETECTING, State.ERROR],
            State.DETECTING: [State.TRACKING, State.ERROR],
            State.TRACKING: [State.GRASPING, State.ERROR],
            State.GRASPING: [State.LIFTING, State.ERROR],
            State.LIFTING: [State.READY, State.ERROR]
        }

    def transition_to(self, new_state: State):
        """状态转换"""
        if new_state in self.transitions[self.current_state]:
            old_state = self.current_state
            self.current_state = new_state
            logging.info(f"状态转换: {old_state.name} -> {new_state.name}")
        else:
            logging.error(f"非法状态转换: {self.current_state.name} -> {new_state.name}")

    def update(self, delta_time: float):
        """状态更新"""
        handler = self.state_handlers.get(self.current_state, self._handle_unknown)
        handler(delta_time)

    def _handle_init(self, delta_time: float):
        """初始化状态处理"""
        pass

    def _handle_calibrating(self, delta_time: float):
        """校准状态处理"""
        pass

    # 其他状态处理方法...

# ---------------------------- 用户界面 ----------------------------
class DebugUI:
    """调试用户界面"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.windows = {}
        self._init_ui()

    def _init_ui(self):
        """初始化UI元素"""
        if self.enabled:
            cv2.namedWindow("Main View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Main View", 800, 600)

    def update(self, system_status: Dict, frame_data: Dict = None):
        """更新UI显示"""
        if not self.enabled:
            return

        # 主显示窗口
        if frame_data:
            display_frame = self._overlay_info(frame_data['processed_frame'], system_status)
            cv2.imshow("Main View", display_frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt

    def _overlay_info(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """叠加调试信息"""
        # 实现信息叠加逻辑...
        return frame

    def close(self):
        """关闭所有UI窗口"""
        if self.enabled:
            cv2.destroyAllWindows()

# ---------------------------- 主程序入口 ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能仓储控制系统")
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument('--simulate', action='store_true', help="启用模拟模式")
    args = parser.parse_args()

    system = WarehouseAutomationSystem(args.config)
    system.startup_sequence()

    try:
        system.main_loop()
    except Exception as e:
        logging.critical(f"未处理的异常: {str(e)}")
        system.shutdown()