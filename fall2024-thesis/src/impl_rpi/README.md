# High-Level Control for Romi Robot Implementation

Detailed information about this codebase see Appendix B of the thesis document.

Below listed relevant links and resources used during the development of this codebase.

Pololu Romi robot
- [Pololu Romi 32U4 Control Board User's Guide](https://www.pololu.com/docs/0J69)
- [Building a Raspberry Pi robot with the Romi chassis](https://www.pololu.com/blog/663/building-a-raspberry-pi-robot-with-the-romi-chassis)
MQTT
- [MQTT Python client documentation](https://eclipse.dev/paho/files/paho.mqtt.python/html/index.html)
- [NanoMQ broker documentation](https://nanomq.io/docs/en/latest/)

IMU Integration
- [LSM6DS33 Datasheet](https://www.pololu.com/file/0J1087/LSM6DS33.pdf)
- [Python library for LSM6DS33](https://github.com/DarkSparkAg/MinIMU-9-v5)
- [Estimating Velocity and Position Using Accelerometers](https://www.pololu.com/file/0J587/AN-1007-EstimatingVelocityAndPositionUsingAccelerometers.pdf)
- [C++ and Python library for IMU targeting Raspberry Pi](https://github.com/RPi-Distro/RTIMULib)
- [IMU dead reckoning implementation for ROS](https://github.com/Abekabe/IMU-Dead-Reckoning/tree/master)
- [A good-practices guideline for 6-DoF IMU-based dead-reckoning](https://lcv.fee.unicamp.br/images/BTSym-22-Brasil/papers/paper_057.pdf)
- [BerryIMU](https://github.com/ozzmaker/BerryIMU)
- [IMU dead reckoning C++ implementation](https://gist.github.com/mpkuse/42c9c89507fd158be310bd7af98db335)
- [INS algorithm using quaternion model for low cost IMU](https://doi.org/10.1016/j.robot.2004.02.001)
- [IMU dead reckoning Medium article](https://towardsdatascience.com/dead-reckoning-is-still-alive-8d8264f7bdee)
- [IMU odomerty for ROS](https://github.com/nadiawangberg/imu_to_odom)

Filtering and Sensor Fusion
- [EKF interative tutorial archive](https://simondlevy.academic.wlu.edu/files/kalman_tutorial/kalman.pdf)
- [Lightweight EKF library in C/C++](https://github.com/simondlevy/TinyEKF)
- [Implementing a Sensor Fusion Algorithm for 3D Orientation Detection with Inertial/Magnetic Sensors](https://www.researchgate.net/publication/264707640_Implementing_a_Sensor_Fusion_Algorithm_for_3D_Orientation_Detection_with_InertialMagnetic_Sensors)
- [Implementing Positioning Algorithms Using Accelerometers](https://www.nxp.com/docs/en/application-note/AN3397.pdf)
- [C sensor fusion library for IMU](https://github.com/xioTechnologies/Fusion)
- [ROS robot_localization source code](https://github.com/cra-ros-pkg/robot_localization)