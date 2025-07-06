# voice_assistant.py
import os
import json
import time
import threading
import requests
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from queue import Queue

# 语音服务配置
CONFIG_PATH = "config/voice_config.json"
DEFAULT_CONFIG = {
    "baidu_api": {
        "app_id": "6869245",
        "api_key": "ZQ3ZxB70xR19hrHM6JWQ8ZKb",
        "secret_key": "aus6ciZdCWBzdOvoStDoOaq3IfwxZajT"
    },
    "voice_params": {
        "spd": 4,    # 语速 (0-9)
        "pit": 8,    # 音调 (0-9)
        "vol": 5,    # 音量 (0-15)
        "per": 5118  # 度小童音色
    },
    "wake_word": "小诺",
    "command_timeout": 5.0,
    "sample_rate": 16000
}

class VoiceAssistant:
    """智能语音助手小诺"""
    def __init__(self):
        self._load_config()
        self._init_audio()
        self.command_queue = Queue()
        self.is_listening = False
        self.is_responding = False
        self._event = threading.Event()
        self.worker_thread = None

    def _load_config(self):
        """加载语音配置"""
        try:
            with open(CONFIG_PATH) as f:
                self.config = json.load(f)
        except:
            self.config = DEFAULT_CONFIG
            logging.warning("使用默认语音配置")

    def _init_audio(self):
        """初始化音频设备"""
        self.audio_buffer = np.array([], dtype=np.int16)
        self.stream = sd.InputStream(
            samplerate=self.config['sample_rate'],
            channels=1,
            callback=self._audio_callback
        )

    def start(self):
        """启动语音助手"""
        self.is_listening = True
        self.worker_thread = threading.Thread(target=self._process_audio)
        self.worker_thread.start()
        logging.info("语音助手小诺已启动")

    def stop(self):
        """停止语音助手"""
        self.is_listening = False
        self._event.set()
        if self.worker_thread:
            self.worker_thread.join()
        logging.info("语音助手已停止")

    def _audio_callback(self, indata, frames, time, status):
        """音频采集回调"""
        if self.is_listening:
            self.audio_buffer = np.append(self.audio_buffer, indata[:,0])

    def _process_audio(self):
        """音频处理线程"""
        while self.is_listening:
            if len(self.audio_buffer) >= self.config['sample_rate'] * 2:  # 处理2秒片段
                self._detect_wake_word()
                self.audio_buffer = np.array([], dtype=np.int16)
            time.sleep(0.1)

    def _detect_wake_word(self):
        """唤醒词检测"""
        temp_file = "temp_audio.wav"
        wavfile.write(temp_file, self.config['sample_rate'], self.audio_buffer)
        
        # 使用百度语音唤醒服务
        result = self._baidu_asr(temp_file)
        if self.config['wake_word'] in result:
            self._respond("在的，请指示")
            self._listen_command()

    def _listen_command(self):
        """进入命令监听模式"""
        self.is_responding = True
        start_time = time.time()
        
        while time.time() - start_time < self.config['command_timeout']:
            audio_data = self._record_command()
            command = self._baidu_asr(audio_data)
            if command:
                self._process_command(command)
                break
        self.is_responding = False

    def _record_command(self, duration=3):
        """录制语音命令"""
        recording = sd.rec(int(duration * self.config['sample_rate']),
                          samplerate=self.config['sample_rate'],
                          channels=1)
        sd.wait()
        return recording.flatten()

    def _process_command(self, command):
        """处理语音命令"""
        if "开始送货" in command:
            self.command_queue.put("start_delivery")
            self._respond("好的，马上开始送货任务")
        elif "停" in command:
            self.command_queue.put("emergency_stop")
            self._respond("正在停止所有操作")
        # 添加更多命令处理...

    def _respond(self, text):
        """语音响应"""
        tts_file = self._baidu_tts(text)
        self._play_audio(tts_file)

    def _baidu_asr(self, audio_data):
        """百度语音识别"""
        url = "https://vop.baidu.com/server_api"
        headers = {
            'Content-Type': 'audio/wav; rate=16000'
        }
        params = {
            'dev_pid': 1537,  # 普通话输入
            'cuid': 'xiaonuo_robot',
            'token': self._get_baidu_token()
        }
        
        try:
            response = requests.post(url, params=params, headers=headers, data=audio_data)
            result = response.json()
            return result['result'][0] if 'result' in result else ""
        except Exception as e:
            logging.error(f"语音识别失败: {str(e)}")
            return ""

    def _baidu_tts(self, text):
        """百度语音合成"""
        url = "https://tsn.baidu.com/text2audio"
        params = {
            'tex': text,
            'tok': self._get_baidu_token(),
            'cuid': 'xiaonuo_robot',
            'ctp': 1,
            'lan': 'zh',
            **self.config['voice_params']
        }
        
        try:
            response = requests.post(url, params=params)
            with open("temp_tts.mp3", "wb") as f:
                f.write(response.content)
            return "temp_tts.mp3"
        except Exception as e:
            logging.error(f"语音合成失败: {str(e)}")
            return None

    def _get_baidu_token(self):
        """获取百度访问令牌"""
        auth_url = "https://openapi.baidu.com/oauth/2.0/token"
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.config['baidu_api']['api_key'],
            'client_secret': self.config['baidu_api']['secret_key']
        }
        response = requests.get(auth_url, params=params)
        return response.json().get('access_token', '')

    def _play_audio(self, file_path):
        """播放音频"""
        if not file_path: return
        
        try:
            fs, data = wavfile.read(file_path)
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            logging.error(f"音频播放失败: {str(e)}")

    def status_report(self, message):
        """状态播报"""
        if not self.is_responding:
            self._respond(message)

class DeliveryAnnouncer(threading.Thread):
    """配送状态自动播报器"""
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self._running = True
        self.announce_interval = 30  # 秒

    def run(self):
        while self._running:
            self.assistant.status_report("小诺正在完成配送任务，请让一下")
            time.sleep(self.announce_interval)

    def stop(self):
        self._running = False