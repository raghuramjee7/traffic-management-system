from detect import *

classes_path = "coco.names"
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"


detector = Detector()
detector.read_classes(classes_path)
detector.download_model(model_url)
detector.load_model()

video_path = "walking.avi"

detector.predict_video(video_path)
