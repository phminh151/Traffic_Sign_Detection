import cv2
from helpers.yolo import load_image, load_yolo, detect_objects, get_box_dimensions, draw_labels
net, classes, colors, output_layers = load_yolo()
img, height, width, channels = load_image('./images/test.jpg')
blob, outputs = detect_objects(img, net, output_layers)
boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
draw_labels(boxes, confs, colors, class_ids, classes, img)
# cv2.imshow("Image", img)
# cv2.waitKey()
# crop_img = img[y:y+h, x:x+w]
		