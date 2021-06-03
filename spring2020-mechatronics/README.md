# Intro to Mechatronics Lab
Offered by Dr. Zachariah Fuchs. See [class detail in Coursicle](https://www.coursicle.com/uc/courses/EECE/5144C/).

## Labs
Files in this directory are solutions to lab sessions.

The hardware used in the lab is ATmega328p embedded on Arduino Uno board, programmed with [USBtinyISP](https://learn.adafruit.com/usbtinyisp).
Please note that flashing the MCU may need extra setup depending on the programmer chip ([for example](https://www.asensar.com/guide/arduino_atemlstudio/integrate-avrdude-with-atmel-studio.html)).

The context of the lab is described in `lab-handout.pdf` under any subdirectories.

All sub-directories are natively project folders in [Atmel Studio (now Microchip Studio)](https://www.microchip.com/en-us/development-tools-tools-and-software/microchip-studio-for-avr-and-sam-devices).
As of now, Atmel Studio is not available on Linux machines, but you may use my [Makefile](https://github.com/liu2z2/mgms/tree/main/src/make-avr-template) for AVR-GCC in another project (this is not tested however).