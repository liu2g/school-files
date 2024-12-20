#include <servo32u4.h>
#include <wpi-32u4-lib.h>
#include <Chassis.h>
#include <Romi32U4.h>
#include "PololuRPiSlave.h"

// https://github.com/pololu/pololu-rpi-slave-arduino-library/blob/master/src/PololuTWISlave.cpp

// keep this under 64 bytes
struct Data                                    // Nth byte
{
  bool yellow, green, red;                     // 0, 1, 2
  bool buttonA, buttonB, buttonC;              // 3, 4, 5

  float leftMotor, rightMotor;                 // 6-9, 10-13
  uint16_t batteryMillivolts;                  // 14-15
  uint16_t analog[6];                          // 16-27
  int16_t leftEncoder, rightEncoder;           // 28-29, 30-31
  uint16_t leftLineSensor, rightLineSensor;    // 32-33, 34-35
  uint16_t servoPercent;                       // 36-37
};

PololuRPiSlave<struct Data,5> slave;
Chassis chassis(7.0, 1440, 14.9); // wheel diam, encoder counts, wheel track
Romi32U4ButtonA buttonA;
Romi32U4ButtonB buttonB;
Romi32U4ButtonC buttonC;
Servo32U4Pin5 servo;

void setup()
{
  // Set up the slave at I2C address 20.
  slave.init(20);
  digitalWrite(4,HIGH);

  servo.attach();
  servo.writeMicroseconds(2000);
  chassis.init();
  chassis.setMotorPIDcoeffs(5, 0.5);
  pinMode(LEFT_LINE_SENSE, INPUT);
  pinMode(RIGHT_LINE_SENSE, INPUT);
}

void loop()
{
  /**************************************************************************
   * Start message buffer
   *************************************************************************/
  slave.updateBuffer();

  /**************************************************************************
   * Read from robot, write to message buffer
   *************************************************************************/
  
  // Buttons
  slave.buffer.buttonA = buttonA.isPressed();
  slave.buffer.buttonB = buttonB.isPressed();
  slave.buffer.buttonC = buttonC.isPressed();

  // Battery voltage
  slave.buffer.batteryMillivolts = readBatteryMillivolts();

  // Motor encoder counts
  slave.buffer.leftEncoder = chassis.getLeftEncoderCount();
  slave.buffer.rightEncoder = chassis.getRightEncoderCount();

  // Misc analog readings
  for(uint8_t i=0; i<6; i++)
  {
    slave.buffer.analog[i] = analogRead(i);
  }
  
  // Line sensors
  slave.buffer.leftLineSensor = static_cast<uint16_t>(analogRead(LEFT_LINE_SENSE));
  slave.buffer.rightLineSensor = static_cast<uint16_t>(analogRead(RIGHT_LINE_SENSE));

  /**************************************************************************
   * Read from message buffer, write to robot
   *************************************************************************/

  // LEDs
  ledYellow(slave.buffer.yellow);
  ledGreen(slave.buffer.green);
  ledRed(slave.buffer.red);

  // Motor target speeds
  chassis.setWheelSpeeds(slave.buffer.leftMotor, slave.buffer.rightMotor);
  
  // Servo for robot arm
  servo.writeMicroseconds(slave.buffer.servoPercent * 10 + 1000);  // 1000-2000 us
  
  /**************************************************************************
   * End message buffer
   *************************************************************************/
  slave.finalizeWrites();
}