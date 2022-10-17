import tensorflow as tf
import os
import time 
import numpy as np
import cv2
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector():

	def __init__(self):
		return

	def read_classes(self, class_file_path):
		with open(class_file_path, 'r') as f:
			self.classes_list = f.read().splitlines()

		# colors for each class in the class list with 3 channels
		l = len(self.classes_list)
		self.color_list = np.random.uniform(low=0,high=255,size=(l,3))

		#print(len(self.classes_list),len(self.color_list))

	def download_model(self, model_url):

		# get model name with model.tar.gz
		model_name = os.path.basename(model_url)

		# extract model name without extensions
		self.model = model_name[:model_name.index('.')]

		# create a cache directory to store our model
		self.cache_dir = "./pretrained_models"
		os.makedirs(self.cache_dir, exist_ok = True)

		# let us download the model
		get_file(fname=model_name, origin=model_url, cache_dir=self.cache_dir,
			cache_subdir="checkpoints", extract = True)

	def load_model(self):
		print("Loading the model " + self.model)
		tf.keras.backend.clear_session()
		self.model_current = tf.saved_model.load(os.path.join(self.cache_dir,
			"checkpoints", self.model, "saved_model"))
		print(self.model + " loaded succesfully...")

	def bounding_box(self, image):
		inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
		inputTensor = inputTensor[tf.newaxis,...]

		detections = self.model_current(inputTensor)
		boxes = detections['detection_boxes'][0].numpy()

		class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
		class_scores = detections['detection_scores'][0].numpy()

		imH, imW, imC = image.shape

		bboxidx = tf.image.non_max_suppression(boxes, class_scores, 
			max_output_size = 50, iou_threshold = 0.5, score_threshold=0.5)

		if len(bboxidx)!=0:
			for i in bboxidx:
				box = tuple(boxes[i].tolist())
				class_confidence = round(class_scores[i]*100)
				class_index = class_indexes[i]

				class_label_text = self.classes_list[class_index]
				class_color = self.color_list[class_index]

				display_text = str(class_label_text) + " " + str(class_confidence)

				ymin, xmin, ymax, xmax = box
				xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
				xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = class_color, thickness = 1)
				cv2.putText(image, display_text, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)
		return image

	def predict_video(self, video_path):
		cap = cv2.VideoCapture(video_path)

		if (cap.isOpened()==False):
			print("Error in opening")
			return

		(success, image) = cap.read()

		start_time = 0

		while success:

			curret_time = time.time()
			fps = 1/(curret_time-start_time)
			start_time = curret_time

			bboximage = self.bounding_box(image)

			cv2.imshow("Result", bboximage)

			key = cv2.waitKey(1) & 0xFF

			if key==ord('q'): break

			(success, image) = cap.read()

		cv2.destroyAllWindows()





