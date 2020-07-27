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

/*! \brief 打印帮助命令
 */
//void ShowHelp(void)
//{
//    printf("\n\r——————————————野火步进电机驱动演示程序——————————————");
//    printf("\n\r输入命令：");
//    printf("\n\r< ? >       -帮助菜单");
//    printf("\n\ra[data]     -设置步进电机的加速度（范围：71—32000）单位为：0.01rad/s^2");
//    printf("\n\rd[data]     -设置步进电机的减速度（范围：71—32000）单位为：0.01rad/s^2");
//    printf("\n\rs[data]     -设置步进电机的最大速度（范围：12—3000）单位为：0.01rad/s");
//    printf("\n\rm[data]     -以设定的步数移动步进电机（范围：-2147483647—2147483647，其中正数为顺时针，负数为逆时针）");
//  
//    printf("\n\rmove [steps] [accel] [decel] [speed]");
//    printf("\n\r            -以设定的步数、加速度，减速度，最大速度开始运动");
//    printf("\n\r<Enter键>   -重复最后一次动作");
//    printf("\n\r开发板按键，KEY1：驱动器禁止输出(脱机状态)  KEY2：驱动器恢复正常工作\n\r ");
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



///*! \brief 打印电机参数
// *  \param acceleration 加速度
// *  \param deceleration 减速度
// *  \param speed        最大速度
// *  \param steps        移动步数
// */
//void ShowData(int position, int acceleration, int deceleration, int speed, int steps)
//{
//  printf("\n\r加速度:%.2frad/s^2",1.0*acceleration/100);
//  printf("  减速度:%.2frad/s^2",1.0*deceleration/100);
//  printf("  最大速度:%.2frad/s(%.2frpm)",1.0*speed/100,9.55*speed/100);
//  printf("  移动步数:%d",steps);
//  printf("\n\r电机当前位置: %d\r\n",position);
//}
/**

  * @brief  处理串口接收到的数据

  * @param  无

  * @retval 无

  */
void DealSerialData(void)
{
    static char showflag =1;
    //默认移动步数
    static int steps = 40000;
    //默认加速度
    static int acceleration = 32000;
    //默认减速度
    static int deceleration = 32000;
    //默认最大速度
    static int speed = 3000;
    
    int acc_temp=0;
    int dec_temp=0;
    int speed_temp=0;
    
    //接收到正确的指令才为TRUE
    char okCmd = FALSE;
    if(showflag)
    {
      showflag = 0;
      ShowData(stepPosition, acceleration, deceleration, speed, steps);
    }
    //检查是否接收到指令
    if(status.cmd == TRUE)
    {        
      if(UART_RxBuffer[0] == 'm')
      {
        //以固定步数移动
        if(UART_RxBuffer[1] == ' ')
        {
          //从串口获取步数
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
              //设置步数、加速度、减速度、最大速度
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
        //设置加速度
        if(UART_RxBuffer[1] == ' ')
        {
          acc_temp = atoi((char const *)UART_RxBuffer+2);
          if(acc_temp>=71 && acc_temp<=32000)
          {
            acceleration = acc_temp;
            //printf("\n\r加速度:%.2frad/s^2",1.0*acceleration/100);
						Usart_SendString(DEBUG_USART, "\n\r  ");
            okCmd = TRUE;
          }
        }
      }
      else if(UART_RxBuffer[0] == 'd')
      {
        //设置减速度
        if(UART_RxBuffer[1] == ' ')
        {
          dec_temp = atoi((char const *)UART_RxBuffer+2);
          if(dec_temp>=71 && dec_temp<=32000)
          {
            deceleration = dec_temp;
            //printf("\n\r减速度:%.2frad/s^2",1.0*deceleration/100);
            okCmd = TRUE;
          }
        }
      }
      else if(UART_RxBuffer[0] == 's')
      {
        //设置最大速度
        if(UART_RxBuffer[1] == ' ')
        {
          speed_temp = atoi((char const *)UART_RxBuffer+2);
          if(speed_temp>=12 && speed_temp<=20000)
          {
            speed = speed_temp;
            //printf("\n\r最大速度:%.2frad/s",1.0*speed/100);
						Usart_SendString(DEBUG_USART, "\n\r  ");
            okCmd = TRUE;
          }
        }
      }
      else if(UART_RxBuffer[0] == 13)
      {
        //如果是回车键直接重复上一次运动
        MSD_Move(steps, acceleration, deceleration, speed);
        okCmd = TRUE;
      }
      else if(UART_RxBuffer[0] == '?')
      {
        //打印帮助命令
        ShowHelp();
        okCmd = TRUE;
      }
      else if(UART_RxBuffer[0] == 't')
      {
        //步进电机测试指令
        MSD_demo_run();         
        okCmd = TRUE;
      }
      //如果指令有无则打印帮助命令
      if(okCmd != TRUE)
      {
        //printf("\n\r 输入有误，请重新输入...");
        Usart_SendString(DEBUG_USART, "Error! Pleace enter again");
				ShowHelp();
      }

      //清空串口接收缓冲数组
      status.cmd = FALSE;
      uart_FlushRxBuffer();

      if(status.running == TRUE)
      {
        if(status.out_ena == TRUE)
				{
            //printf("\n\r电机正在运行...");
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

  * @brief  驱动器运行演示

  * @param  无

  * @retval 无

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

void Delay(__IO uint32_t nCount)	 //简单的延时函数

{

	for(; nCount != 0; nCount--);

}
/*********************************************END OF FILE**********************/
