import cv2
import os
import numpy
def facesetting():
  haar_file = "haarcascade_frontalface_default.xml"
  datasets = "face"
  sub_data = input("Enter the name of the person: ").strip()

  path = os.path.join(datasets, sub_data)
  os.makedirs(path, exist_ok=True)


  face_cascade = cv2.CascadeClassifier(haar_file)
  webcam = cv2.VideoCapture(0)


  if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

  count = 1
  while count <= 50:
    print(f"Capturing image {count}")
    
    
    ret, im = webcam.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

 
    for (x, y, w, h) in faces:
        
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

      
        cv2.putText(im, f"Capturing image {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

       
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (100, 100))

        cv2.imwrite(os.path.join(path, f"{count}.png"), face_resize)
        count += 1

    cv2.imshow("Face Detection", im)

    key = cv2.waitKey(1)
    if key == ord("q"):
        print("Exiting...")
        break


  webcam.release()
  cv2.destroyAllWindows()
def checkingimage():
   size=4
   haar_file="haarcascade_frontalface_default.xml"
   datasets="face"
   print('Training...')
   (images,labels,names,id)=([],[],{},0)
   for (subdirs,dirs,files) in os.walk(datasets):
      for subdir in dirs:
         names[id]=subdir
         subjectpath=os.path.join(datasets,subdir)
         for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
         id+=1
   (width,height)=(100,100)
   (images, labels) = [numpy.array(lis) for lis in [images, labels]]
   model=cv2.face.LBPHFaceRecognizer_create()
   model.train(images,labels)
   face_cascade=cv2.CascadeClassifier(haar_file)
   webcam=cv2.VideoCapture(0)
   cnt=0
   while True:
      (_,im)=webcam.read()
      gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
      faces=face_cascade.detectMultiScale(gray,1.3,5)
      for(x,y,w,h) in faces:
         cv2.rectangle(im,(x+y),(x+w,y+h),(255,255,0),2)
         face=gray[y:y+h,x:x+w]
         face_resize=cv2.resize(face,(width,height))
         prediction=model.predict(face_resize)
         cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
         if prediction[1]<800:
            cv2.putText(im,"%s-%0f"%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT)
            print(names[prediction[0]])
            cnt=0
         else:
            cnt+=1
            cv2.putText(im,'Unknown',(x-10,y-10),cv2.FONT)
            if(cnt>100):
               print('unknown')
               cv2.imwrite("image.jpg",im)
               cnt=0
      cv2.imshow("opencv",im)
      k=cv2.waitKey(1)
      if k==ord('q'):
         break
   webcam.realease()
   cv2.destroyAllWindows()
n=int(input("enter your choice\n 1.upload your face \n 2.check your presence\n"))
if(n==1):
   facesetting()
elif(n==2):
   checkingimage()