import cv2
import numpy as np

'''
    evaluate image classification using opencv,
    Use opencv to run tensorflow model
'''

net5 = cv2.dnn.readNetFromTensorflow('log/opt_model6.pb')
#Read the image to test
image = cv2.imread('images/2.png')
inputa = cv2.dnn.blobFromImage(image, size=(54,54), swapRB=True, crop=False)
net5.setInput(inputa)

result =''
a5= net5.forward(['digit1/dense/MatMul','digit2/dense/MatMul','digit3/dense/MatMul','digit4/dense/MatMul', 'digit5/dense/MatMul'])
for a in a5:
    aa1 = np.argmax(a)
    if aa1<10:
        result+= str(aa1)

print(result)

