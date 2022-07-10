#include "sapi.h"
#include "arm_math.h"



/* ************************************************************************* */
/*                               Configuration                               */
/* ************************************************************************* */
# define   SIGNAL_10B         0
# define   SIGNAL_4B          1
# define   SIGNAL_ORIGINAL    2

#define PSF_MODE  SIGNAL_10B

uint32_t tick   = 0   ;
uint16_t tone   = 440 ;

/* ************************************************************************* */
/*                                    Code                                   */
/* ************************************************************************* */
struct header_struct {
   char     head[4];
   uint32_t id;
   uint16_t N;
   uint16_t fs ;
   uint32_t maxIndex;
   uint32_t minIndex;
   q15_t    maxValue;
   q15_t    minValue;
   q15_t    rms;
   char     tail[4];
} header={"head",0,128,10000,0,0,0,0,0,"tail"};


int main ( void ) {
   uint16_t sample = 0;
   int16_t adc [ header.N ];
   boardConfig (                  );
   uartConfig  ( UART_USB, 460800 );
   adcConfig   ( ADC_ENABLE       );
   dacConfig   ( DAC_ENABLE       );
   cyclesCounterInit ( EDU_CIAA_NXP_CLOCK_SPEED );
   while(1) {
      cyclesCounterReset();
      float t=(tick/(float)header.fs);
      tick++;
      q15_t og_sample = 512*arm_sin_f32(t*2*PI*tone);
      #if PSF_MODE == SIGNAL_10B
         adc[sample] = (((adcRead(CH1)-512)) << 6);
      #elif PSF_MODE == SIGNAL_4B
         adc[sample] = (((adcRead(CH1)-512)) >> 6 << 12);
      #elif PSF_MODE == SIGNAL_ORIGINAL
         adc[sample] = og_sample << 6;
      #endif
      uartWriteByteArray ( UART_USB ,(uint8_t* )&adc[sample] ,sizeof(adc[0]) );
      dacWrite(DAC, og_sample + 512);
      if ( ++sample==header.N ) {
         gpioToggle ( LEDR ); // este led blinkea a fs/N
         arm_max_q15 ( adc, header.N, &header.maxValue,&header.maxIndex );
         arm_min_q15 ( adc, header.N, &header.minValue,&header.minIndex );
         arm_rms_q15 ( adc, header.N, &header.rms                       );
         //trigger(2);
         header.id++;
         uartWriteByteArray ( UART_USB ,(uint8_t*)&header ,sizeof(header ));
         adcRead(CH1); //why?? hay algun efecto minimo en el 1er sample.. puede ser por el blinkeo de los leds o algo que me corre 10 puntos el primer sample. Con esto se resuelve.. habria que investigar el problema en detalle
         sample = 0;
      }
      gpioToggle ( LED1 );                                           // este led blinkea a fs/2
      while(cyclesCounterRead()< EDU_CIAA_NXP_CLOCK_SPEED/header.fs) // el clk de la CIAA es 204000000
         ;
   }
}

