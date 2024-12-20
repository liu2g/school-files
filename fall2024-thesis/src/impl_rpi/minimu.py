#!venv/bin/python3
# Authored by Caleb G. Teague in 2017
# Copied from https://github.com/DarkSparkAg/MinIMU-9-v5 and modified

import smbus2 as smbus
import time
import math

class MinIMU:
    SMBusNum = 1
    aFullScale = 2
    gFullScale = 500
    mFullScale = 4
    """
    Valid values for aFullScale : 2, 4, 8, and 16 [g]
                     gFullScale : 125, 245, 500, 1000, and 2000 [dps]
                     mFullScale : 4, 8, 12, and 16 [guass]
    """
    def __init__(self):

        #Accelerometer and Gyro Register addresses
        self.Accel_Gyro_REG = dict(
            FUNC_CFG_ACCESS     = 0x01,
                                \
            FIFO_CTRL1          = 0x06,
            FIFO_CTRL2          = 0x07,
            FIFO_CTRL3          = 0x08,
            FIFO_CTRL4          = 0x09,
            FIFO_CTRL5          = 0x0A,
            ORIENT_CFG_G        = 0x0B,
                                \
            INT1_CTRL           = 0x0D,
            INT2_CTRL           = 0x0E,
            WHO_AM_I            = 0x0F,
            CTRL1_XL            = 0x10,
            CTRL2_G             = 0x11,
            CTRL3_C             = 0x12,
            CTRL4_C             = 0x13,
            CTRL5_C             = 0x14,
            CTRL6_C             = 0x15,
            CTRL7_G             = 0x16,
            CTRL8_XL            = 0x17,
            CTRL9_XL            = 0x18,
            CTRL10_C            = 0x19,
                                \
            WAKE_UP_SRC         = 0x1B,
            TAP_SRC             = 0x1C,
            D6D_SRC             = 0x1D,
            STATUS_REG          = 0x1E,
                                \
            OUT_TEMP_L          = 0x20,
            OUT_TEMP_H          = 0x21,
            OUTX_L_G            = 0x22,
            OUTX_H_G            = 0x23,
            OUTY_L_G            = 0x24,
            OUTY_H_G            = 0x25,
            OUTZ_L_G            = 0x26,
            OUTZ_H_G            = 0x27,
            OUTX_L_XL           = 0x28,
            OUTX_H_XL           = 0x29,
            OUTY_L_XL           = 0x2A,
            OUTY_H_XL           = 0x2B,
            OUTZ_L_XL           = 0x2C,
            OUTZ_H_XL           = 0x2D,
                                \
            FIFO_STATUS1        = 0x3A,
            FIFO_STATUS2        = 0x3B,
            FIFO_STATUS3        = 0x3C,
            FIFO_STATUS4        = 0x3D,
            FIFO_DATA_OUT_L     = 0x3E,
            FIFO_DATA_OUT_H     = 0x3F,
            TIMESTAMP0_REG      = 0x40,
            TIMESTAMP1_REG      = 0x41,
            TIMESTAMP2_REG      = 0x42,
                                \
            STEP_TIMESTAMP_L    = 0x49,
            STEP_TIMESTAMP_H    = 0x4A,
            STEP_COUNTER_L      = 0x4B,
            STEP_COUNTER_H      = 0x4C,
                                \
            FUNC_SRC            = 0x53,
                                \
            TAP_CFG             = 0x58,
            TAP_THS_6D          = 0x59,
            INT_DUR2            = 0x5A,
            WAKE_UP_THS         = 0x5B,
            WAKE_UP_DUR         = 0x5C,
            FREE_FALL           = 0x5D,
            MD1_CFG             = 0x5E,
            MD2_CFG             = 0x5F)

        #Magnemometer addresses
        self.Mag_REG= dict(
            WHO_AM_I    = 0x0F,
                        \
            CTRL_REG1   = 0x20,
            CTRL_REG2   = 0x21,
            CTRL_REG3   = 0x22,
            CTRL_REG4   = 0x23,
            CTRL_REG5   = 0x24,
                        \
            STATUS_REG  = 0x27,
            OUT_X_L     = 0x28,
            OUT_X_H     = 0x29,
            OUT_Y_L     = 0x2A,
            OUT_Y_H     = 0x2B,
            OUT_Z_L     = 0x2C,
            OUT_Z_H     = 0x2D,
            TEMP_OUT_L  = 0x2E,
            TEMP_OUT_H  = 0x2F,
            INT_CFG     = 0x30,
            INT_SRC     = 0x31,
            INT_THS_L   = 0x32,
            INT_THS_H   = 0x33)

        #Unit scales
        self.aScale = 0 # default: aScale = 2g/2^15,
        self.gScale = 0 # default: gScale = 500dps/2^15
        self.mScale = 0 # default: mScale = 4guass/2^15

        #Variables for updateAngle and updateYaw
        self.prevAngle = [[0,0,0]] #x, y, z (roll, pitch, yaw)
        self.prevYaw = [0]
        self.tau = 0.04 #Want this roughly 10x the dt
        self.lastTimeAngle = [0]
        self.lastTimeYaw = [0]

        #i2c addresses
        self.mag = 0x1e #0011110 (from docs)
        self.accel_gyro = 0x6b

        #Connect i2c bus
        self.bus = smbus.SMBus(self.SMBusNum)

    def enableAccel_Gyro(self, aFullScale: float = None, gFullScale: float = None):
        """Setup the needed registers for the Accelerometer and Gyro"""
        aFullScale = aFullScale or self.aFullScale
        gFullScale = gFullScale or self.gFullScale

        g = 9.806
        #the gravitational constant for a latitude of 45 degrees at sea level is 9.80665
        #g for altitude is g(6,371.0088 km / (6,371.0088 km + altitude))^2
        #9.80600 is a good approximation for Tulsa, OK

        #default: 0b10000000
        #ODR = 1.66 kHz; +/-2g; BW = 400Hz
        b0_3 = 0b1000 #1.66 kHz

        #full-scale selection; 2**15 = 32768
        if aFullScale == 4:
            b4_5 = 0b10
            self.aScale = 4*g/32768
        elif aFullScale == 8:
            b4_5 = 0b11
            self.aScale = 8*g/32768
        elif aFullScale == 16:
            b4_5 = '01'
            self.aScale = 16*g/32768
        else: #default to 2g if no valid value is given
            b4_5 = '00'
            self.aScale = 2*g/32768

        b6_7 = '00' #0b00; 400Hz anti-aliasing filter bandwidth

        # self.bus.write_byte_data(self.accel_gyro, self.Accel_Gyro_REG['CTRL1_XL'], 0b10000000)
        self.bus.write_byte_data(self.accel_gyro,
                                 self.Accel_Gyro_REG['CTRL1_XL'],
                                 self.binConcat(b0_3, b4_5, b6_7))

        #Gyro

        #default: 0b010000000
        #ODR = 1.66 kHz; 500dps
        b0_3 = 0b1000 #1.66 kHz

        #full-scale selection
        if gFullScale == 254:
            b4_6 = '000'
            self.gScale = 254/32768.0
        elif gFullScale == 1000:
            b4_6 = 0b100
            self.gScale = 1000/32768.0
        elif gFullScale == 2000:
            b4_6 = 0b110
            self.gScale = 2000/32768.0
        elif gFullScale == 125:
            b4_6 = '001'
            self.gScale = 125/32768.0
        else: #default to 500 dps if no valid value is given
            b4_6 = '010'
            self.gScale = 500/32768.0

        # self.bus.write_byte_data(self.accel_gyro, self.Accel_Gyro_REG['CTRL2_G'], 0b10000100)
        self.bus.write_byte_data(self.accel_gyro,
                                 self.Accel_Gyro_REG['CTRL2_G'],
                                 self.binConcat(b0_3, b4_6, 0))

        #Accelerometer and Gyro

        #default: 0b00000100
        #IF_INC = 1 (automatically increment register address)
        self.bus.write_byte_data(self.accel_gyro, self.Accel_Gyro_REG['CTRL3_C'], 0b00000100)

    def enableMag(self, mFullScale = None):
        """Setup the needed registers for the Magnetometer"""
        mFullScale = mFullScale or self.mFullScale

        #default: 0b01110000
        #Temp off, High-Performance, ODR = 300Hz, Self_test off
        self.bus.write_byte_data(self.mag, self.Mag_REG['CTRL_REG1'], 0b01010010)

        #default: 0b00000000
        # +/-4guass, reboot off, soft_reset off

        #full-scale selection; 2**15 = 32768
        if mFullScale == 8:
            b1_2 = '01'
            self.mScale = 8.0/32768
        elif mFullScale == 12:
            b1_2 = 0b10
            self.mScale = 12.0/32768
        elif mFullScale == 16:
            b1_2 = 0b11
            self.mScale = 16.0/32768
        else: #default to 4 guass if no valid value is given
            b1_2 = '00'
            self.mScale = 4.0/32768

        rebootMem = False #Reboot memory content
        softReset = False #Configuration registers and user register reset function

        # self.bus.write_byte_data(self.mag, self.Mag_REG['CTRL_REG2'], 0b00000000)
        self.bus.write_byte_data(self.mag,
                                 self.Mag_REG['CTRL_REG2'],
                                 self.binConcat(0, b1_2, 0, rebootMem, softReset, 0, 0))

        #default: 0b00000011
        #Low-power off, default SPI, continous convo mode
        self.bus.write_byte_data(self.mag, self.Mag_REG['CTRL_REG3'], 0b00000000)

        #default: 0b00000000
        #High-Performance, data LSb at lower address
        self.bus.write_byte_data(self.mag, self.Mag_REG['CTRL_REG4'], 0b00001000)

    def readAccel(self) -> tuple[float, float, float]:
        """Get readings on X, Y, Z axes from accelerometer in m/s^2, nan if unable to read"""

        try:
            # Reading low and high 8-bit register and converting the 16-bit two's complement number to decimal.
            ax = self.byteToNumber(self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTX_L_XL']),
                                    self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTX_H_XL']))

            ay = self.byteToNumber(self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTY_L_XL']),
                                    self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTY_H_XL']))

            az = self.byteToNumber(self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTZ_L_XL']),
                                    self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTZ_H_XL']))
            return ax * self.aScale, ay * self.aScale, az * self.aScale
        except:
            return float("nan"), float("nan"), float("nan")

    def readGyro(self) -> tuple[float, float, float]:
        """Get readings on X, Y, Z axes from gyro in deg/sec, nan if unable to read"""
        try:
            # Reading low and high 8-bit register and converting the 16-bit two's complement number to decimal.
            gx = self.byteToNumber(self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTX_L_G']),
                                    self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTX_H_G']))

            gy = self.byteToNumber(self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTY_L_G']),
                                    self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTY_H_G']))

            gz = self.byteToNumber(self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTZ_L_G']),
                                    self.bus.read_byte_data(self.accel_gyro, self.Accel_Gyro_REG['OUTZ_H_G']))
            return gx * self.gScale, gy * self.gScale, gz * self.gScale
        except:
            return float("nan"), float("nan"), float("nan")

    def readMag(self) -> tuple[float, float, float]:
        """Get readings on X, Y, Z axes from magnetometer in guass, nan if unable to read"""
        try:
            # Reading low and high 8-bit register and converting the 16-bit two's complement number to decimal.
            mx = self.byteToNumber(self.bus.read_byte_data(self.mag, self.Mag_REG['OUT_X_L']), \
                                    self.bus.read_byte_data(self.mag, self.Mag_REG['OUT_X_H']))

            my = self.byteToNumber(self.bus.read_byte_data(self.mag, self.Mag_REG['OUT_Y_L']), \
                                    self.bus.read_byte_data(self.mag, self.Mag_REG['OUT_Y_H']))

            mz = self.byteToNumber(self.bus.read_byte_data(self.mag, self.Mag_REG['OUT_Z_L']), \
                                    self.bus.read_byte_data(self.mag, self.Mag_REG['OUT_Z_H']))
            return mx * self.mScale, my * self.mScale, mz * self.mScale
        except:
            return float("nan"), float("nan"), float("nan")

    def updateAngle(self) -> tuple[float, float, float]:
        """Read from accelerometer, gyro, and magnetometer to get the current roll, pitch, and yaw in deg with a complementaty filter"""
        [ax, ay, az] = self.readAccel()
        [gx_w, gy_w, gz_w] = self.readGyro()
        [mx, my, mz] = self.readMag()

        if self.lastTimeAngle[0] == 0: #If this is the first time using updatePos
            self.lastTimeAngle[0] = time.time()

        #Find the angle change given by the Gyro
        dt = time.time() - self.lastTimeAngle[0]
        gx = self.prevAngle[0][0] + gx_w * dt
        gy = self.prevAngle[0][1] + gy_w * dt
        gz = self.prevAngle[0][2] + gz_w * dt

        #Using the Accelerometer find pitch and roll
        rho = math.degrees(math.atan2(ax, math.sqrt(ay**2 + az**2))) #pitch
        phi = math.degrees(math.atan2(ay, math.sqrt(ax**2 + az**2))) #roll

        #Using the Magnetometer find yaw
        theta = math.degrees(math.atan2(-1*my, mx)) + 180 #yaw

        #To deal with modular angles in a non-modular number system I had to keep
        #the Gz and theta values from 'splitting' where one would read 359 and
        #other 1, causing the filter to go DOWN from 359 to 1 rather than UP.  To
        #accomplish this this I 'cycle' the Gz value around to keep the
        #complementaty filter working.
        if gz - theta > 180:
            gz = gz - 360
        if gz - theta < -180:
            gz = gz + 360

        #This must be used if the device wasn't laid flat
        #theta = math.degrees(math.atan2(-1*My*math.cos(rho) + Mz*math.sin(phi), Mx*math.cos(rho) + My*math.sin(rho)*math.sin(phi) + Mz*math.sin(rho)*math.cos(phi)))

        #This combines a LPF on phi, rho, and theta with a HPF on the Gyro values
        alpha = self.tau/(self.tau + dt)
        xAngle = (alpha * gx) + ((1-alpha) * phi)
        yAngle = (alpha * gy) + ((1-alpha) * rho)
        zAngle = (alpha * gz) + ((1-alpha) * theta)

        #Update previous angle with the current one
        self.prevAngle[0] = [xAngle, yAngle, zAngle]

        #Update time for dt calculations
        self.lastTimeAngle[0] = time.time()

        return xAngle, yAngle, zAngle #roll, pitch, yaw

    def updateYaw(self):
        """Read from gyro, and magnetometer to get the current yaw in deg with a complementaty filter"""
        [gx_w, gy_w, gz_w] = self.readGyro()
        [mx, my, mz] = self.readMag()

        if self.lastTimeYaw[0] == 0: #If this is the first time using updatePos
            self.lastTimeYaw[0] = time.time()

        #Find the angle change given by the Gyro
        dt = time.time() - self.lastTimeYaw[0]
        gz = self.prevYaw[0] + gz_w * dt

        #Using the Magnetometer find yaw
        theta = math.degrees(math.atan2(-1*my, mx)) + 180 #yaw

        #To deal with modular angles in a non-modular number system I had to keep
        #the Gz and theta values from 'splitting' where one would read 359 and
        #other 1, causing the filter to go DOWN from 359 to 1 rather than UP.  To
        #accomplish this this I 'cycle' the Gz value around to keep the
        #complementaty filter working.
        if gz - theta > 180:
            gz = gz - 360
        if gz - theta < -180:
            gz = gz + 360

        #This combines a LPF on phi, rho, and theta with a HPF on the Gyro values
        alpha = self.tau/(self.tau + dt)
        zAngle = (alpha * gz) + ((1-alpha) * theta)

        #Update previous yaw with the current one
        self.prevYaw[0] = zAngle

        #Update time for dt calculations
        self.lastTimeYaw[0] = time.time()

        return zAngle

    @staticmethod
    def byteToNumber(low: int, high: int):
        """Combines high and low 8-bit values to a 16-bit two's complement and converts to decimal"""
        number = 256 * high + low #2^8 = 256
        if number >= 32768: #2^7 = 32768
            number= number - 65536 #For two's complement
        return number

    @staticmethod
    def binConcat(*values):
        """Concatonate a list of values (integer, boolean, or string) into a single binary number"""
        strValue = '0b' + ''.join([x if isinstance(x, str) else bin(x)[2:] for x in values])
        return int(strValue, 2)


def main():
        imu = MinIMU()
        imu.enableAccel_Gyro()

        while True:
            print("A:" + ", ".join("{:.2E}".format(x) for x in imu.readAccel()))
            print("G:" + ", ".join("{:.2E}".format(x) for x in imu.readGyro()))
            time.sleep(1)

if __name__ == "__main__":
    main()
