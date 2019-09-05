import cv2,time,pandas
from datetime import datetime

first_frame=None#a special type of data type which allows you to create a variable and assign nothing to it but we the variable there.
#when we call this variable later python will not say variable is not defined
status_list=[None,None] #as statement is looking for 2 items and its item later on 0 is added but still there is no 2nd item so we added none.
times=[]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)#its a method that triggers a video capture obj. if we are capturing through a camera or built in cam we pass number as an argument.
#if we have multiple cams we pass index as 0,1,2 eg.(0),we passn the video file name

while True:
    #check is bool data type and frame is numpy array
    check, frame=video.read()
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0) #we use to make the image blurry and to smooth it remove nois n detaile and increase accuracy in the calculation
                                #tuple where with, height of the gaussian kernal

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray) #absolute difference between first frame and the current frame
                    #here first frame is blurry gary version and 2nd frame is gray image
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]#we get a black and white picture using binary
                                                            #1 is written to access the second item from the tuple
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)#here bigger the number soother the image

    #different syntax #to check the area of countors #to draw the external countors of the object that we will be finding in the image.
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#approximation method that opencv will apply for retriving the contours
            #the arguments are the frame where i want to find the countours and use the copy of it.

    for contour in cnts:
        if cv2.contourArea(contour)<30000:
            continue
        status=1

        (x,y,w,h)=cv2.boundingRect(contour)#defines the parameters of rectangle and what would be equal to
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)

    status_list=status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        if len(times) % 2 != 0:                 # check if "times" list is odd
            del times[0]
        break

print(status_list)
print(times)
    #datetime.now
    #it will record the time of each frame
for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()#to relase your cam or to start it.
cv2.destroyAllWindows
