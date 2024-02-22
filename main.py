import cv2
import numpy as np
import RPi.GPIO as GPIO

# Define the GPIO pin and the function to drive the car


PWMA = 18  # PWMA를 GPIO 18에 연결
PWMB = 23  # PWMB를 GPIO 23에 연결
AIN1 = 22  # AIN1을 GPIO 22에 연결
AIN2 = 27  # AIN2을 GPIO 27에 연결
BIN1 = 24  # BIN1을 GPIO 24에 연결
BIN2 = 25  # BIN2을 GPIO 25에 연결

GPIO.setmode(GPIO.BCM)

GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)


# 좌측 및 우측 모터를 제어하기 위한 PWM 객체 생성 및 시작
L_Motor = GPIO.PWM(PWMA, 500)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB, 500)
R_Motor.start(0)

# needed function: forward, left, right, stop
def forward(speed):
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(speed)
    
def left(speed):
    GPIO.output(AIN1,1)
    GPIO.output(AIN2,0)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(speed)
    
def right(speed):
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN1,1)
    GPIO.output(BIN2,0)
    R_Motor.ChangeDutyCycle(speed)

def stop():
    GPIO.output(AIN1,0)
    GPIO.output(AIN2,1)
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN1,0)
    GPIO.output(BIN2,1)
    R_Motor.ChangeDutyCycle(0)


def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)
    
    # 빨간색, 노란색 및 파란색에 해당하는 HSV 범위 정의
    lower_red = np.array([119, 65, 39])
    upper_red = np.array([255, 255, 255])

    lower_yellow = np.array([16, 76, 93])
    upper_yellow = np.array([255, 255, 255])

    lower_blue = np.array([91, 210, 13])
    upper_blue = np.array([255, 255, 255])
    
    
    
    
    frame_width = 640

    while camera.isOpened():
         success, image = camera.read() #image is in the form of 3d matrix
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

       
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        
        combined_mask = mask_red | mask_yellow | mask_blue 
        
        result = cv2.bitwise_and(image, image, mask=combined_mask)
        
        # get the contours of the masked image 윤곽을 추출
        # definition of contours: a curve that joining all continuous points having the same colour or intensity
        
        # masking is used to define zero and non-zero pixels. (0~255로 정의)
        
        # if the pixel is in the lower and upper boundary of the mask, set pixel to true(255, 255, 255), else false(0,0,0)
        # get the contours of the masked image
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.findContours함수 : combined_mask에서 이미지를 찾고,contours변수에 윤곽을 저장 

    
        # after you get all the contours available from the masked image, loop through
        # all the contours and select the contours with the largest area, e.g. has the most amount of non-zero pixels : contours(윤곽)중에서 가장 면적이 큰 윤곽을 선택
        max_area = 0
        max_contour = None
        pixels = np.sum(combined_mask >0) # 0이 아닌 픽셀 수 계산
        
        for c in contours:
            # get the area of each contour
            area = cv2.contourArea(c)            
        # to eliminate contours with too few amount of non-zero pixels, create a condition: 적은양의 픽셀을 제외
        # if contour area is larger than previously captured contour area AND contour area has more than 2000 non-zero pixels 새로운 면적 > 이전캡쳐한 윤곽
        # update our max_area and max_contour variable 현재픽셀 > 2000개 이상의 non픽셀(흰색픽셀)    
            if area > max_area and area > 2000:
                max_area = area
                max_contour = c

		
        # Once you get the contours which has the largest area with the most amount of pixels,
        # take the bounding rectangle of the maximum contour
        x,y,w,h = 0,0,0,0
        if max_contour is not None:
            x,y,w,h = cv2.boundingRect(max_contour)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
   
                    
        # from the bounding rectangle, take the x,y origin and the height,width of the bounding rectangle
        # we can then take the center position of the bounding rectangle (called centroid)  
        # and make it as the coordinates for our car to keep track of it
        # fill your code here
            centroid_x = x+w//2
            centroid_y = y+h//2
        # you can also create a circle that corresponds to the center of the bounding rectangle
        cv2.circle(image, (centroid_x,centroid_y), 10, (0, 0, 255), -1)

		# display your resulted image
        cv2.imshow('output video',image)   
        
		# Here lies the condition in which you need to check whether the centroid is located
		# in the frame. If the centroid is located in the middle, then go forward
		# if centroid is located in the left, then turn left
		# if centroid is located in the right, then turn right
		# note that in the beginning, in the camera.set() function, we have defined the height and width of our captured frame
		# Use this as your reference to define the middle area, left area, and right area of your frame
		# complete the if else statement below.
        if centroid_x < frame_width /3:
            print("turn left")
            left(50)
        elif frame_width /3 <= centroid_x < 2 * frame_width/3:
            print("go straight")
            forward(50)
        	# in order to make the car dont hit the colord object we are tracking, we can just calculate the amount of pixels.
        	# if the pixel is larger than some amount, make the car stop
        	# Hint: from the resulted mask, calculate the pixel that is non-zero
			# fill your code here
            if pixels > 5000:
                print("stop!")
                stop()
          
        elif 2* frame_width /3 <= centroid_x:
            print("go right")
            right(50)
        else:
            print("cannot detect target")
            stop()

		# function used to end the program
        if cv2.waitKey(1) == ord('q'):
            print("quiting")
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
