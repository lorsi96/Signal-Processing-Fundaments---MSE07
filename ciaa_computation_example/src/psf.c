#include "sapi.h"
#include "arm_math.h"



q7_t q7_multiply(q15_t a, q15_t b) {
   q15_t res = (a * b) << 1; // q14 -> q15.
   return (q7_t) (res >> 8); // Trunco.
}


uint16_t q7_print(q7_t n, char *buf)
{
   int i;
   float ans=(n&0x80)?-1:0;
   for(i=1;i<8;i++)
   {
      if(n&(0x80>>i)){
         ans+=1.0/(1U<<i);
      }
   }
   return sprintf(buf,"q7: 0x%x (%i dec); float:%.20f\r\n", n, n, ans);
}


int main ( void ) {
   uint16_t sample = 0;
   int16_t len;
   char buf [200] = {0};

   boardConfig (                  );
   uartConfig  ( UART_USB, 460800 );
   adcConfig   ( ADC_ENABLE       );
   cyclesCounterInit ( EDU_CIAA_NXP_CLOCK_SPEED );

   q7_t a = 0x040;
   q7_t b = 0x023;
   
   for(;;) {
      cyclesCounterReset();
      len = q7_print(q7_multiply(a, b), buf);
      uartWriteByteArray (UART_USB, buf ,len);
      gpioToggle (LED1);
      delay(500);
   }
}
