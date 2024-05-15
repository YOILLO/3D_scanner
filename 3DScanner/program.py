from socket import *
from time import *
import pyurg
from math import *
from GPIO import *
import Neural

addr = ("192.168.43.200", 9090)

GPIO_init()

tcp_socket = socket(AF_INET,  SOCK_STREAM)

urg = pyurg.UrgDevice()

if not urg.connect():
    print('Could not connect.')
    exit()

tcp_socket.connect(addr)

motor_step = 0
scan_num = 0
cloud = []
cloud_prev = []
file = open(f"scans/{scan_num}.txt")
neural = Neural.onnxCNN(200, 700)
while True:
	try:
		print("Trying connect")
		tcp_socket.connect(addr)
		tcp_socket.recv(1)

		while True:
			if motor_step > 125:
				file.close()
				if len(cloud_prev):
					file_transform = open("transforms/" + scan_num + ".txt")
					while len(cloud) < 700:
						cloud.append([0, 0, 0])
					file_transform.write(neural.forward(cloud_prev, cloud))
					file_transform.close()
				cloud_prev = cloud
				cloud = []
				scan_num += 1
				file = open("scans/" + scan_num + ".txt")
				motor_step = 0
				#tcp_socket.send(b"STOP\n")
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

				cloud.append([x, y, z])

				stt = str(int(round(x))) + " " + str(int(round(y))) + " " + str(int(round(z))) + "\n"
				file.write(stt)
				#print(stt)
				#tcp_socket.send(bytes(stt))
			GPIO_step2()
			motor_step += 1
	except:
		print("Error")
	finally:
		sleep(10)
