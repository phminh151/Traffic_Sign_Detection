from helpers.yolo import load_yolo, detect_objects, get_box_dimensions, draw_labels_video, webcam_detect
import cv2
import time
model, classes, colors, output_layers = load_yolo()
		
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

starting_time = time.time()
frame_id = 0

# Start detecting...
while True:

    # Read a video frame
    _, frame = cap.read()
    frame_id += 1

    height, width, channel = frame.shape
    # blob = cv2.dnn.blobFromImage(
    #     frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    blob, outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels_video(boxes, confs, colors, class_ids, classes, frame)
    
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    # Show FPS
    cv2.putText(frame, "FPS: " + str(round(fps, 2)),
                (10, 50), font, 2, (0, 0, 0), 3)

    # Display image
    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break
