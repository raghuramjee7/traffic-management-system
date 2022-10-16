# import human detection, vehilce detetion classes here

from vehicle_detection import VehicleDetector
from file_upload import *

import threading
import time

def create_map():

	j1 = Junction()
	j2 = Junction()
	j3 = Junction()
	j4 = Junction()
	j5 = Junction()

	j1.front = j2
	j1.left = j5

	j2.front = j3
	j2.back = j1
	j2.left = j5

	j3.back = j2
	j3.left = j4

	j4.back = j5
	j4.left = j3

	j5.front = j4
	j5.back = j1
	j5.right = j2

	return [j1, j2, j3, j4]


class Junction():

	def __init__(self,):

		self.vehicle_detection = VehicleDetector()
		self.human_detection = None
		self.left = None
		self.right = None
		self.front = None
		self.back = None
		return

jx = VehicleDetector()
link = get_link_j1()

t1 = threading.Thread(target=jx.detect, args=(link,))
t1.start()
state = True
for i in range(1000):
	time.sleep(3)
	print("cars", jx.car_count)
	state = t1.is_alive()
	if state==False: break
if state==False: t1.join()

