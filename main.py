

from ultralytics import YOLO
import cv2

# from sort.sort import *
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)
# tracker = Sort()
cocomodel = YOLO("yolov8n.pt")
license_plate_detector = YOLO("license_plate_detector.pt")
# cap = cv2.VideoCapture("sample.mp4")
cap = cv2.imread("ebest.jpg")


 # Close the window

foundIt = False
detections = cocomodel(cap)[0]
detections_list = []
license_plate_list = []
license_plate_text = []
for detection in detections.boxes.data.tolist():
    
    x1,y1,w,h,score,class_id = detection 
    detections_list.append([int(x1),int(y1),int(w),int(h)])
 


license_plates = license_plate_detector(cap)[0]

for license_plate in license_plates.boxes.data.tolist():

    lx1,ly1,lw,lh,lscore,lclass_id = license_plate
    license_plate_list.append([int(lx1), int(ly1), int(lw),  int(lh)])
    
    print(license_plate_list)


for j in range(len(detections_list)):
        xcar1, ycar1, xcar2, ycar2 = detections_list[j]
        for i in range(len(license_plate_list)):
            lx1, ly1, lw, lh = license_plate_list[i]
           
            if int(lx1) > xcar1 and int(ly1) > ycar1 and int(lw) < xcar2 and int(lh) < ycar2:
                car_indx = j
                x1,y1,w,h = detections_list[car_indx]
                # print("this is the license plate of car :",car_indx)
                print(int(ly1),int(h),int(lx1),int(lw))
                croped_license_plate = cap[int(ly1):int(lh), int(lx1):int(lw), : ]


                print(croped_license_plate.shape)
                gray_scaled_licensePlate =  cv2.cvtColor(croped_license_plate,cv2.COLOR_BGR2GRAY)
                _, gray_scaled_licensePlate_tresh =cv2.threshold(gray_scaled_licensePlate,64,255,cv2.THRESH_BINARY_INV)
             
                
                magnification_factor = 4

                # Get the original dimensions of the cropped image
                original_height, original_width, _ = croped_license_plate.shape

                # Calculate the new dimensions after magnification
                new_height = int(original_height * magnification_factor)
                new_width = int(original_width * magnification_factor)

                # Magnify the image using OpenCV's resize function
                magnified_image = cv2.resize(gray_scaled_licensePlate, (new_width, new_height))

                detections = reader.readtext(magnified_image)
                k = 0
                # text = pytesseract.image_to_string(gray_scaled_licensePlate, lang='eng', output_type=pytesseract.Output.DICT)
                # print(text)
                # license_plate_text = []
                # text = text["text"].upper().replace(' ', '')
                # license_plate_text.append(text)
                

                # for i in range(len(detections['text'])):
                #     text = detections['text'][i].upper().replace(' ', '')
                #     confidence = int(detections['conf'][i])

                #     # Filter out non-empty text and low-confidence detections
                #     if text != '' and confidence > 0:
                #         license_plate_text.append((text, confidence))

                # for text, score in license_plate_text:
                #      print(text, score)
                for detection in detections:
                    
                    bbox, text, score = detection

                    text = text.upper().replace(' ', '')
                    license_plate_text.append(text)
                    print(text, score)

                cv2.rectangle(cap, (int(x1), int(y1)), ( int(w),  int(h)), (0, 255, 0), 2)
                cv2.rectangle(cap, (int(lx1), int(ly1)), ( int(lw),  int(lh)), (0, 0, 255), 2)
                print(license_plate_text)
                cv2.putText(cap , text, (lx1-80,lh-100), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0,0,255), thickness =3)

                break
            



cv2.imshow("cars",cap)  # Display the image in a window named 'Loaded Image'
cv2.waitKey(0)  # Wait until a key is pressed (0 means indefinite wait)
cv2.destroyAllWindows() 


print("all good")