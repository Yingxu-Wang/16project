// Lifting Platform Control
#include <AccelStepper.h>

// 电机定义
#define MOTOR1_STEP 2
#define MOTOR1_DIR 3
#define MOTOR2_STEP 4
#define MOTOR2_DIR 5
#define MOTOR3_STEP 6
#define MOTOR3_DIR 7

AccelStepper liftMotor1(AccelStepper::DRIVER, MOTOR1_STEP, MOTOR1_DIR);
AccelStepper liftMotor2(AccelStepper::DRIVER, MOTOR2_STEP, MOTOR2_DIR);
AccelStepper pushMotor(AccelStepper::DRIVER, MOTOR3_STEP, MOTOR3_DIR);

// 运行参数
const long LAYER_TIMES[] = {0, 2000, 3500, 5000}; // 各层对应运行时间（毫秒）
const long PUSH_DURATION = 1000; // 推送电机运行时间

enum State { IDLE, LIFTING, PUSHING, RETURNING };
State currentState = IDLE;
unsigned long actionStartTime = 0;
int targetLayer = 0;

void setup() {
  Serial.begin(115200);
  
  // 初始化电机参数
  liftMotor1.setMaxSpeed(1000);
  liftMotor1.setAcceleration(500);
  liftMotor2.setMaxSpeed(1000);
  liftMotor2.setAcceleration(500);
  pushMotor.setMaxSpeed(2000);
  pushMotor.setAcceleration(1000);

  pinMode(8, INPUT_PULLUP); // 限位开关
}

void loop() {
  switch(currentState){
    case IDLE:
      processSerialCommands();
      break;
      
    case LIFTING:
      if(millis() - actionStartTime >= LAYER_TIMES[targetLayer]){
        stopMotors();
        currentState = PUSHING;
        actionStartTime = millis();
        // 启动推送电机
        pushMotor.moveTo(2000);
      }
      break;
      
    case PUSHING:
      pushMotor.run();
      if(!pushMotor.isRunning()){
        // 返回推送电机
        pushMotor.moveTo(0);
        currentState = RETURNING;
      }
      break;
      
    case RETURNING:
      pushMotor.run();
      if(!pushMotor.isRunning()){
        // 反转升降电机
        liftMotor1.setSpeed(-500);
        liftMotor2.setSpeed(-500);
        actionStartTime = millis();
      }
      
      // 检测原点位置
      if(digitalRead(8) == LOW){
        stopMotors();
        currentState = IDLE;
        Serial.println("ACK");
      }
      break;
  }
}

void processSerialCommands(){
  if(Serial.available()){
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if(cmd.startsWith("LIFT:")){
      targetLayer = cmd.substring(5).toInt();
      if(targetLayer >=1 && targetLayer <=3){
        Serial.println("ACK");
        startLifting();
      }else{
        Serial.println("ERROR:INVALID_LAYER");
      }
    }
  }
}

void startLifting(){
  // 启动升降电机
  liftMotor1.setSpeed(500);
  liftMotor2.setSpeed(500);
  actionStartTime = millis();
  currentState = LIFTING;
}

void stopMotors(){
  liftMotor1.stop();
  liftMotor2.stop();
  liftMotor1.runToPosition();
  liftMotor2.runToPosition();
}