/*
 * Lab3_0.c
 *
 * Created: 2/12/2020 16:41:13
 * Author : leo85
 */ 

#include <avr/io.h>
#include <avr/interrupt.h>

/* Pin config
PD0   : Spinner switch
PB0-1 : Digit0-1 select
PB2   : Segment A
PC5-0 : Segment B-G
*/

volatile int last_pin; //Store the pin value from last iteration
volatile int edge_count; //Store the number of falling edges on the switch
volatile int seg_sel; //Flag to select two 7 segs
volatile int seg_bin = 0; //7 bit var to store 7 seg on/off values
int bin_to_segs (int);

//Interrupt to toggle two 7 seg selects b/w 01 and 10
ISR(TIMER1_COMPA_vect)
{
	if (seg_sel) {
		PORTB |= (1 << PORTB0); //Deselect first 7 seg
		PORTB &= ~(1 << PORTB1); //Select second 7 seg
		seg_bin = bin_to_segs( edge_count & 0b1111 );
		//PORTC = (~PIND) & 0b111111;
	}
	else {
		PORTB &= ~(1 << PORTB0); //Select first 7 seg
		PORTB |= (1 << PORTB1); //Deselect second 7 seg
		seg_bin = bin_to_segs( (edge_count >> 4) & 0b1111 );
		//PORTC = (~PIND) & 0b111111;
	}
	
	//Turn on/off seg A
	if (seg_bin >> 6) PORTB |= (1 << PORTB2);
	else PORTB &= ~(1 << PORTB2);
	
	//Turn on/off seg B-G
	PORTC = seg_bin & 0b111111;
	
	//Toggle flag
	seg_sel ^= 1;
}

int main(void)
{
	//Pin config
	DDRD = 0b11111110;
	DDRB = 0b00000111;
	DDRC = 0b00111111;
	

	//Timer and interrupt config
	cli();
	TCCR1A = (0<<COM1A1)|(0<<COM1A0)|(0<<COM1B1)|(0<<COM1B0)|(0<<WGM11)|(0<<WGM10); //Timer1 CTC
	TCCR1B = (0<<ICNC1)|(0<<ICES1)|(0<<WGM13)|(1<<WGM12)|(1<<CS12)|(0<<CS11)|(0<<CS10); //Timer1 scaled by 256
	TIMSK1 = (0<<ICIE1)|(0<<OCIE1B)|(1<<OCIE1A)|(0<<TOIE1); //Enable match interrupt
	//OCR1A = 62500; //1 sec
	//OCR1A = 15625; //250 ms
	//OCR1A = 6250;  //100 ms
	//OCR1A = 1875; //30 ms
	OCR1A = 62; //1 ms
	sei(); //Enable global interrupt
	
	
    while (1) 
    {
		if ((last_pin == 1) && (PIND == 0)) edge_count += 1;
		last_pin = PIND;
		//while (PIND) {}
		//edge_count += 1;
    }
}

int bin_to_segs (int bin_value) //Converts 4 bit value from switches to 7 bit A-G segs
{
	switch (bin_value)
	{
		case 0x0:
			return 0b1111110;
			break;
		case 0x1:
			return 0b0110000;
			break;
		case 0x2:
			return 0b1101101;
			break;
		case 0x3:
			return 0b1111001;
			break;
		case 0x4:
			return 0b0110011;
			break;
		case 0x5:
			return 0b1011011;
			break;
		case 0x6:
			return 0b1011111;
			break;
		case 0x7:
			return 0b1110000;
			break;
		case 0x8:
			return 0b1111111;
			break;
		case 0x9:
			return 0b1110011;
			break;
		case 0xA:
			return 0b1110111;
			break;
		case 0xb:
			return 0b0011111;
			break;
		case 0xC:
			return 0b1001110;
			break;
		case 0xd:
			return 0b0111101;
			break;
		case 0xE:
			return 0b1001111;
			break;
		case 0xF:
			return 0b1000111;
			break;
		default:
			return 0b0000000;
				
	}
}

