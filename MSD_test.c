/**
  ******************************************************************************
  * @file    MSD_test.c
  * @author  fire
  * @version V1.0
  * @date    2013-xx-xx
  * @brief   MSD���Դ���
  ******************************************************************************
  * @attention
  *
  * ʵ��ƽ̨:Ұ��STM32 F429-��ս�� ������  
  * ��̳    :http://www.firebbs.cn
  * �Ա�    :http://fire-stm32.taobao.com
  *
  ******************************************************************************
  */ 
#include "MSD_test.h"  
#include <math.h>
#include <stdlib.h>
#include "./usart/bsp_debug_usart.h"
#include "./key/bsp_exti.h" 
#include "./led/bsp_led.h"
#include "MicroStepDriver.h" 
void ShowHelp(void);
void ShowData(int position, int acceleration, int deceleration, int speed, int steps);
void Delay(__IO u32 nCount);

/*! \brief ��ӡ��������
 */
//void ShowHelp(void)
//{
//    printf("\n\r����������������������������Ұ�𲽽����������ʾ���򡪡�������������������������");
//    printf("\n\r�������");
//    printf("\n\r< ? >       -�����˵�");
//    printf("\n\ra[data]     -���ò�������ļ��ٶȣ���Χ��71��32000����λΪ��0.01rad/s^2");
//    printf("\n\rd[data]     -���ò�������ļ��ٶȣ���Χ��71��32000����λΪ��0.01rad/s^2");
//    printf("\n\rs[data]     -���ò������������ٶȣ���Χ��12��3000����λΪ��0.01rad/s");
//    printf("\n\rm[data]     -���趨�Ĳ����ƶ������������Χ��-2147483647��2147483647����������Ϊ˳ʱ�룬����Ϊ��ʱ�룩");
//  
//    printf("\n\rmove [steps] [accel] [decel] [speed]");
//    printf("\n\r            -���趨�Ĳ��������ٶȣ����ٶȣ�����ٶȿ�ʼ�˶�");
//    printf("\n\r<Enter��>   -�ظ����һ�ζ���");
//    printf("\n\r�����尴����KEY1����������ֹ���(�ѻ�״̬)  KEY2���������ָ���������\n\r ");
//    
//}

void ShowData(int position, int acceleration, int deceleration, int speed, int steps)
{
  Usart_SendString(DEBUG_USART, "\n\r  Motor pos: ");
  Usart_SendInt(DEBUG_USART, position);
  Usart_SendString(DEBUG_USART, "    a:");
	Usart_SendInt(DEBUG_USART, (acceleration));
  Usart_SendString(DEBUG_USART, "  d:");
  Usart_SendInt(DEBUG_USART, (deceleration));
  Usart_SendString(DEBUG_USART, "  s:");
  Usart_SendInt(DEBUG_USART, (speed));
  Usart_SendString(DEBUG_USART, "  m:");
  Usart_SendInt(DEBUG_USART, (steps));
  Usart_SendString(DEBUG_USART, "\n\r> ");
	
}

static const char Help[] = {"\n\r--------------------------------------------------------------\n\r - Linear speed control of stepper motor\n\r\n\r?        - Show help\n\ra [data] - Set acceleration (range: 71 - 32000)\n\rd [data] - Set deceleration (range: 71 - 32000)\n\rs [data] - Set speed (range: 12 - motor limit)\n\rm [data] - Move [data] steps (range: -64000 - 64000)\n\rmove [steps] [accel] [decel] [speed]\n\r         - Move with all parameters given\n\r<enter>  - Repeat last move\n\r\n\r    acc/dec data given in 0.01*rad/sec^2 (100 = 1 rad/sec^2)\n\r    speed data given in 0.01*rad/sec (100 = 1 rad/sec)\n\r--------------------------------------------------------------\n\r"};

/*! \brief Sends help message.
 *
 *  Outputs help message.
 */
void ShowHelp(void)
{
  unsigned int i = 0;
  while(Help[i] != 0)
    Usart_SendByte(DEBUG_USART,Help[i++]);
}



///*! \brief ��ӡ�������
// *  \param acceleration ���ٶ�
// *  \param deceleration ���ٶ�
// *  \param speed        ����ٶ�
// *  \param steps        �ƶ�����
// */
//void ShowData(int position, int acceleration, int deceleration, int speed, int steps)
//{
//  printf("\n\r���ٶ�:%.2frad/s^2",1.0*acceleration/100);
//  printf("  ���ٶ�:%.2frad/s^2",1.0*deceleration/100);
//  printf("  ����ٶ�:%.2frad/s(%.2frpm)",1.0*speed/100,9.55*speed/100);
//  printf("  �ƶ�����:%d",steps);
//  printf("\n\r�����ǰλ��: %d\r\n",position);
//}
/**

  * @brief  �����ڽ��յ�������

  * @param  ��

  * @retval ��

  */
void DealSerialData(void)
{
    static char showflag =1;
    //Ĭ���ƶ�����
    static int steps = 40000;
    //Ĭ�ϼ��ٶ�
    static int acceleration = 32000;
    //Ĭ�ϼ��ٶ�
    static int deceleration = 32000;
    //Ĭ������ٶ�
    static int speed = 3000;
    
    int acc_temp=0;
    int dec_temp=0;
    int speed_temp=0;
    
    //���յ���ȷ��ָ���ΪTRUE
    char okCmd = FALSE;
    if(showflag)
    {
      showflag = 0;
      ShowData(stepPosition, acceleration, deceleration, speed, steps);
    }
    //����Ƿ���յ�ָ��
    if(status.cmd == TRUE)
    {        
      if(UART_RxBuffer[0] == 'm')
      {
        //�Թ̶������ƶ�
        if(UART_RxBuffer[1] == ' ')
        {
          //�Ӵ��ڻ�ȡ����
          steps = atoi((char const *)UART_RxBuffer+2);
          MSD_Move(steps, acceleration, deceleration, speed);
          okCmd = TRUE;
          //printf("\n\r  ");
					Usart_SendString(DEBUG_USART, "\n\r  ");
					
        }
        else if(UART_RxBuffer[1] == 'o')
        {
          if(UART_RxBuffer[2] == 'v')
          {
            if(UART_RxBuffer[3] == 'e')
            {
              //���ò��������ٶȡ����ٶȡ�����ٶ�
              if(UART_RxBuffer[4] == ' ')
              {
                int i = 6;
                steps = atoi((char const *)UART_RxBuffer+5);
                while((UART_RxBuffer[i] != ' ') && (UART_RxBuffer[i] != 13)) i++;
                i++;
                acceleration = atoi((char const *)UART_RxBuffer+i);
                while((UART_RxBuffer[i] != ' ') && (UART_RxBuffer[i] != 13)) i++;
                i++;
                deceleration = atoi((char const *)UART_RxBuffer+i);
                while((UART_RxBuffer[i] != ' ') && (UART_RxBuffer[i] != 13)) i++;
                i++;
                speed = atoi((char const *)UART_RxBuffer+i);
                MSD_Move(steps, acceleration, deceleration, speed);
                okCmd = TRUE;
                //printf("\n\r  ");
								Usart_SendString(DEBUG_USART, "\n\r  ");
              }
            }
          }
        }
      }
      else if(UART_RxBuffer[0] == 'a')
      {
        //���ü��ٶ�
        if(UART_RxBuffer[1] == ' ')
        {
          acc_temp = atoi((char const *)UART_RxBuffer+2);
          if(acc_temp>=71 && acc_temp<=32000)
          {
            acceleration = acc_temp;
            //printf("\n\r���ٶ�:%.2frad/s^2",1.0*acceleration/100);
						Usart_SendString(DEBUG_USART, "\n\r  ");
            okCmd = TRUE;
          }
        }
      }
      else if(UART_RxBuffer[0] == 'd')
      {
        //���ü��ٶ�
        if(UART_RxBuffer[1] == ' ')
        {
          dec_temp = atoi((char const *)UART_RxBuffer+2);
          if(dec_temp>=71 && dec_temp<=32000)
          {
            deceleration = dec_temp;
            //printf("\n\r���ٶ�:%.2frad/s^2",1.0*deceleration/100);
            okCmd = TRUE;
          }
        }
      }
      else if(UART_RxBuffer[0] == 's')
      {
        //��������ٶ�
        if(UART_RxBuffer[1] == ' ')
        {
          speed_temp = atoi((char const *)UART_RxBuffer+2);
          if(speed_temp>=12 && speed_temp<=20000)
          {
            speed = speed_temp;
            //printf("\n\r����ٶ�:%.2frad/s",1.0*speed/100);
						Usart_SendString(DEBUG_USART, "\n\r  ");
            okCmd = TRUE;
          }
        }
      }
      else if(UART_RxBuffer[0] == 13)
      {
        //����ǻس���ֱ���ظ���һ���˶�
        MSD_Move(steps, acceleration, deceleration, speed);
        okCmd = TRUE;
      }
      else if(UART_RxBuffer[0] == '?')
      {
        //��ӡ��������
        ShowHelp();
        okCmd = TRUE;
      }
      else if(UART_RxBuffer[0] == 't')
      {
        //�����������ָ��
        MSD_demo_run();         
        okCmd = TRUE;
      }
      //���ָ���������ӡ��������
      if(okCmd != TRUE)
      {
        //printf("\n\r ������������������...");
        Usart_SendString(DEBUG_USART, "Error! Pleace enter again");
				ShowHelp();
      }

      //��մ��ڽ��ջ�������
      status.cmd = FALSE;
      uart_FlushRxBuffer();

      if(status.running == TRUE)
      {
        if(status.out_ena == TRUE)
				{
            //printf("\n\r�����������...");
					  Usart_SendString(DEBUG_USART, "Stepper Running");
				}   
        while(status.running == TRUE)
        {
          if(status.out_ena != TRUE)
          break;
        };
        if(status.out_ena == TRUE)
        {
          //printf("OK\n\r");
					Usart_SendString(DEBUG_USART, "OK\n\r  ");
          ShowData(stepPosition, acceleration, deceleration, speed, steps);  
        }

      }

      
    }//end if(cmd)
}
/**

  * @brief  ������������ʾ

  * @param  ��

  * @retval ��

  */
void MSD_demo_run(void)
{
    uint8_t step_cnt=0;
    int8_t step_num[10] = {2,2,-2,-2,2,2,-4,-4,4,20};
    for(int i=0;i<200;i++)
    {
            if(step_cnt==11)
                step_cnt=0;
            while(status.running == TRUE)
            {
                if(status.out_ena != TRUE)
                   break;
            };
            if(status.out_ena != TRUE)
                break;
            else
            {
                Delay(0xFFFFF);
                MSD_Move(SPR*step_num[step_cnt], 32000, 32000, 3000);

                ShowData(stepPosition, 32000, 32000, 3000, SPR*step_num[step_cnt]);
                step_cnt++;  
            }
    }
    step_cnt=0;
}

void Delay(__IO uint32_t nCount)	 //�򵥵���ʱ����

{

	for(; nCount != 0; nCount--);

}
/*********************************************END OF FILE**********************/
