#include <avr/io.h>
#include <avr/interrupt.h>

volatile int seg_bin;
int bin_to_segs (int);

int main(void)
{
	//Pin config
	DDRD = 0;
	DDRB = 0b00000100;
	DDRC = 0b00111111;
	
	
    while (1) 
    {
		seg_bin = bin_to_segs( (~PIND) & 0b1111 );
		//Turn on/off seg A
		if (seg_bin >> 6) PORTB |= (1 << PORTB2);
		else PORTB &= ~(1 << PORTB2);
	
		//Turn on/off seg B-G
		PORTC = seg_bin & 0b111111;

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