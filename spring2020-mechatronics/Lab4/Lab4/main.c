/*
 * Lab4.c
 *
 * Created: 2/17/2020 17:46:26
 * Author : leo85
 */ 

#include <avr/io.h>
#include <avr/interrupt.h>

/*Pinouts
PD1:to strobe RCK 0->1->0
PD0:to strobe SH  1->0->1
PD2:Debug LED
PC5:debug LED
PB5:SPI SCK
PB3:SPI MOSI
PB2:SPI SS
*/

volatile uint8_t stored_data;
//volatile uint8_t rw;

ISR(SPI_STC_vect) //SPI Serial Transfer Complete 
{
	//Strobe Reg2
	PORTD|=(1<<PORTD1);
	PORTD&=~(1<<PORTD1);
	
	//Read
	stored_data = SPDR;
	
	//Strobe Reg1
	PORTD&=~(1<<PORTD0);
	PORTD|=(1<<PORTD0);
	
	//Write
	SPDR = stored_data;
	
}


int main(void)
{
    
	//Pin config
	DDRD=0b00000111;
	DDRB=0b00101100;
	DDRC=0b00100000;
	
	//Enable SPI in interrupt, MSB first, master, mode#0, f/128
	SPCR=(1<<SPIE)|(1<<SPE)|(0<<DORD)|(1<<MSTR)|(0<<CPOL)|(0<<CPHA)|(1<<SPR1)|(1<<SPR0);
	
	//Timer and interrupt config
	cli();
	TCCR1A = (0<<COM1A1)|(0<<COM1A0)|(0<<COM1B1)|(0<<COM1B0)|(0<<WGM11)|(0<<WGM10); //Timer1 CTC
	TCCR1B = (0<<ICNC1)|(0<<ICES1)|(0<<WGM13)|(1<<WGM12)|(1<<CS12)|(0<<CS11)|(0<<CS10); //Timer1 scaled by 256
	TIMSK1 = (0<<ICIE1)|(0<<OCIE1B)|(1<<OCIE1A)|(0<<TOIE1); //Enable match interrupt
	OCR1A = 62500; //1 sec
	sei(); //Enable global interrupt
	
	SPDR = stored_data;
	
	while (1) {}
}

