from ultralytics import YOLO
import cv2
import cvzone
import math
    
if __name__ == "__main__":
    
    model = YOLO("yolov8n.pt")
    capture = cv2.VideoCapture(0)
    
    while True:
        flag, frame = capture.read()
        if not flag:
            exit(1)
        
        result = model(frame, classes = 0, verbose = False)[0]
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            
            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]

            cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(35, y1)), scale = 0.5)
        
        cv2.imshow('Camera Feed', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()