import serial
import time
serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

peel1 = 'm 00001@'
peel2 = 'm 05000@'


for i in range(8):
    serial_port.write(peel2[i].encode())
    print(peel1[i])
    #  由于Jetson Nano本身串口问题
    # 发送一整串的数据就会导致单片机接收到的都是乱码
    # 所以采用间隔0.1s的方式确保信息发送成功
    time.sleep(0.1)



