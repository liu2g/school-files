/*
 * Lab5_slave.c
 *
 * Created: 2/24/2020 18:28:12
 * Author : leo85
 */ 

#include <avr/io.h>
#include <avr/interrupt.h>

volatile uint16_t score;//Score variable
volatile uint8_t packcount; //Counter of UART packet 

int main(void)
{
	//Configure Pins
	DDRC = (1<<PORTC5)|(1<<PORTC4)|(1<<PORTC3)|(1<<PORTC2)|(1<<PORTC1)|(1<<PORTC0);
	DDRD = (1<<PORTD2)|(1<<PORTD3)|(1<<PORTD4)|(1<<PORTD6)|(1<<PORTD7);
	//*****************************************************************************
	//Configure Timer 1
	//*****************************************************************************
	//Toggle OCCR1A on Compare, run in CTC (Clear Timer on Compare)
	TCCR1A = (0<<COM1A1)|(0<<COM1A0)|(0<<WGM11)|(0<<WGM10);
	//CTC Mode, CLK/256  
	TCCR1B = (0<<WGM13)|(1<<WGM12)|(1<<CS12)|(0<<CS11)|(0<<CS10);
	TIMSK1 = (1<<OCIE1A);
	OCR1A = 62;//  (1s)*(16 Mhz/256)

	//*****************************************************************************
	//Configure USART
	//*****************************************************************************
	UCSR0A = (0<<U2X0)|(1<<MPCM0); //Normal speed, multi-processor mode
	UCSR0B = (1<<RXCIE0)|(1<<RXEN0)|(1<<UCSZ02); //RX Complete Interrupt Enabled, Receiver Enabled
	UCSR0C = (0<<UMSEL00)|(0<<UMSEL01)|(0<<UPM00)|(0<<UPM01)|(0<<USBS0)|(1<<UCSZ01)|(1<<UCSZ00); //Async, no parity, 1 stop bit, 9-bit size, 250k Baud
	UBRR0L = 3;
	UBRR0H = 0;
	
	

	//Enable all interrupts
	sei();
    while(1)
    {
        //Do Nothing
    }
}

ISR(USART_RX_vect){
	if (RXB80 && UDR0 == 0x10) {  //1st packet uses multiprocessor and 9th bit
			UCSR0A &= ~(1<<MPCM0);
			packcount ++;		
	}
	if (MPCM0 == 0) { //#2-6 packets do not use multiprocessor
		switch (packcount) {
			case 1:
				if (UDR0 == 0x1) packcount++; //Verify lower part of data by addr
				break;
			case 2:
				score = UDR0;  //Keep first part of data
				packcount++;
				break;
			case 3:
				if (UDR0 == 0x2) packcount++; //Verify higher part of data by addr
				break;
			case 4:
				score += UDR0 << 8; //Cat higher part of data 
				packcount ++;
				break;
			case 5:
				if (UDR0 == 0xFF) { //Verify end of transmission
					packcount = 0;
					UCSR0A |= (1<<MPCM0); //Change back to multiprocessor
				}
				break;
		}
	}
}

ISR(TIMER1_COMPA_vect)
{
	static uint8_t digit_index = 0;//Keep track of which digit is activated
	static uint8_t digit = 0;//Value to display for current digit


	PORTD = 0b10100011|(digit_index<<2);//Activate Appropriate Digit
	
	switch (digit_index)//Calculate value to display for current digit
	{
		case 4:
			digit = score % 10; break;//Ones
		case 3:
			digit = (score % 100)/10; break;//Tens
		case 2:
			digit = (score %1000)/100; break;//Hundreds
		case 1:
			digit = (score %10000)/1000; break;//Thousands
		case 0:
			digit = score/10000; break;//Ten Thousands
		default:
			digit = 0;	
			
	}
	
	switch (digit)//Display character for current digit
	{
		case 0:
				PORTD = (PORTD&0x7F)|0x00;
				PORTC = (PORTC&0xC0)|0x3F; break;
		case 1:
				PORTD = (PORTD&0x7F)|0x00;
				PORTC = (PORTC&0xC0)|0x06; break;
		case 2:
				PORTD = (PORTD&0x7F)|0x80;
				PORTC = (PORTC&0xC0)|0x1B; break;
		case 3:
				PORTD = (PORTD&0x7F)|0x80;
				PORTC = (PORTC&0xC0)|0x0F; break;
		case 4:
				PORTD = (PORTD&0x7F)|0x80;
				PORTC = (PORTC&0xC0)|0x26; break;
		case 5:
				PORTD = (PORTD&0x7F)|0x80;
				PORTC = (PORTC&0xC0)|0x2D; break;
		case 6:
				PORTD = (PORTD&0x7F)|0x80;
				PORTC = (PORTC&0xC0)|0x3D; break;
		case 7:
				PORTD = (PORTD&0x7F)|0x00;
				PORTC = (PORTC&0xC0)|0x07; break;
		case 8:
				PORTD = (PORTD&0x7F)|0x80;;
				PORTC = (PORTC&0xC0)|0x3F; break;
		case 9:
				PORTD = (PORTD&0x7F)|0x80;
				PORTC = (PORTC&0xC0)|0x27; break;
		default:
				PORTD = (PORTD&0x7F);
				PORTC = (PORTC&0xC0)|0x3F; break;
	}
	
	//Updated digit_index for next digit
	if(digit_index++>3){
		digit_index=0;}
	
}