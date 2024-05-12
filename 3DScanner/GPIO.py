from time import *
import OPi.GPIO as GPIO

def GPIO_init():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(18, GPIO.OUT)

def GPIO_step1():
	GPIO.output(18, True)

def GPIO_step2():
	GPIO.output(18, False)
