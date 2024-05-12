from socket import *
from time import *
import pyurg
from math import *
from GPIO import *

addr = ("192.168.43.200", 9090)

GPIO_init()

tcp_socket = socket(AF_INET,  SOCK_STREAM)

urg = pyurg.UrgDevice()

if not urg.connect():
    print('Could not connect.')
    exit()

#tcp_socket.connect(addr)

motor_step = 0
while True:
	try:
		print("Trying connect")
		tcp_socket.connect(addr)

		while True:
			if motor_step > 125:
				motor_step = 0
				tcp_socket.send(b"STOP\n")
				print("STOP")

			data, timestamp = urg.capture()
			GPIO_step1()

			print(motor_step)

			size_of_one_motor_step = 2*3.141592/250
			angle_horiz = motor_step * size_of_one_motor_step

			size_of_one_step = 4.18879/len(data)
			#print len(data)

			for masure_num in range(len(data)):
				angle_vert = size_of_one_step * masure_num + 1.0472

				r = data[masure_num]
				if r <= 10:
					continue
				x = r * sin(angle_vert) * cos(angle_horiz)
				y = r * sin(angle_vert) * sin(angle_horiz)
				z = r * cos(angle_vert)

				stt = str(int(round(x))) + " " + str(int(round(y))) + " " + str(int(round(z))) + "\n"
				#print(stt)
				tcp_socket.send(bytes(stt))
			GPIO_step2()
			motor_step += 1
	except:
		print("Error")
	finally:
		sleep(10)
