#include "sapi.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "fir.h" 

#define BITS    10   // cantidad de bits usado para cuantizar

int16_t offset = 512;
int16_t zero = 0;

struct header_struct {
   char     pre[8];
   uint32_t id;
   uint16_t N;
   uint16_t fs ;
   uint16_t hLength ;
   char     pos[4];
} __attribute__ ((packed));

struct header_struct header={"*header*",0,128,8000,h_LENGTH,"end*"};

int main ( void ) {
   uint16_t sample = 0;
   int16_t adc   [ header.N            ];
   int16_t y     [ h_LENGTH+header.N-1 ];

   boardConfig();
   uartConfig(UART_USB, 460800);
   adcConfig (ADC_ENABLE);
   dacConfig(DAC_ENABLE);
   cyclesCounterInit (EDU_CIAA_NXP_CLOCK_SPEED);

   for(;;) {
      cyclesCounterReset();
      adc[sample] = (((int16_t )adcRead(CH1)-512)>>(10-BITS))<<(6+10-BITS);
      dacWrite(DAC, y[sample]); // will be 128 samples delayed from input.
      if (++sample==header.N ) {
         gpioToggle ( LEDR );
         sample = 0;
         arm_conv_q15(adc, header.N, h, h_LENGTH, y);
         header.id++;
         uartWriteByteArray(
            UART_USB,
            (uint8_t*)&header,
            sizeof(struct header_struct)
         );
         for (int i=0;i<(header.N+h_LENGTH-1); i++) {
            uartWriteByteArray (
               UART_USB,
               (uint8_t*)(i<header.N ? &adc[i] : &offset),
               sizeof(adc[0])
            );
            uartWriteByteArray(
               UART_USB,
               (uint8_t* )(i<h_LENGTH ? &h[i] : &zero),
               sizeof(h[0])
            );
            uartWriteByteArray(
               UART_USB,
               (uint8_t*)(&y[i]),
               sizeof(y[0])
            );
         }
         adcRead(CH1);
      }
      gpioToggle(LED1);
      while(cyclesCounterRead() < EDU_CIAA_NXP_CLOCK_SPEED / header.fs)
         ;
   }
}
