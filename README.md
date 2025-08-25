# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** PRASANNA V
- **Register Number:** 212223240123

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img_bgr =cv2.imread('Eagle_in_Flight.jpg',0)
```

#### 2. Print the image width, height & Channel.
```python
img_bgr.shape
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img_bgr,cmap='gray')
plt.title('BGR Image')
plt.axis('on')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
img=cv2.imread('Eagle_in_Flight.jpg')
cv2.imwrite('Eagle.png',img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(img_rgb)
plt.title("RGB IMAGE")
plt.axis("on")  
plt.show()
img_rgb.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped=img_rgb[20:420,200:550]
plt.imshow(cropped[:,:,::-1])
plt.title("CROPPED IMAGE")
plt.axis("off")
plt.show()
```

#### 8. Resize the image up by a factor of 2x.
```python
res= cv2.resize(cropped,(200*2, 200*2))
```

#### 9. Flip the cropped/resized image horizontally.
```python
flipped_img = cv2.flip(cropped, 1)
plt.imshow(flipped_img[:,:,::-1])
plt.title('Flipped Horizontal')
plt.axis('off')
plt.show()
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img=cv2.imread('Apollo-11-launch.jpg',cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = cv2.putText(img_rgb, "Apollo 11 Saturn V Launch, July 16, 1969", (300, 700),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
plt.imshow(text, cmap='gray')  
plt.title("TEXT IMAGE")
plt.show()  
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rcol= (255, 0, 255)
cv2.rectangle(img_rgb, (400, 100), (800, 650), rcol, 3)
```

#### 13. Display the final annotated image.
```python
plt.title("Annotated image")
plt.imshow(img_rgb)
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
img =cv2.imread('boy.jpg',cv2.IMREAD_COLOR)
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
```

#### 15. Adjust the brightness of the image.
```python
m = np.ones(img_rgb.shape, dtype="uint8") * 50
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img_rgb, m)  
img_darker = cv2.subtract(img_rgb, m)  
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(img_brighter), plt.title("Brighter Image"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(img_darker), plt.title("Darker Image"), plt.axis("off")
plt.show()
```

#### 18. Modify the image contrast.
```python
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape,) * 1.2
img_lower = np.uint8(cv2.multiply(np.float64(img_rgb),matrix1))
img_higher = np.uint8(cv2.multiply(np.float64(img_rgb),matrix2))
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=[18,5])
plt.subplot(131), plt.imshow(img_rgb), plt.title("Original Image"), plt.axis("off")
plt.subplot(132), plt.imshow(img_lower), plt.title("Lower Contrast"), plt.axis("off")
plt.subplot(133), plt.imshow(img_higher), plt.title("Higher Contrast"), plt.axis("off")
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(b), plt.title("Blue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(g), plt.title("Green Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(r), plt.title("Red Channel"), plt.axis("off")
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merged_rgb = cv2.merge([r, g, b])
plt.figure(figsize=(5,5))
plt.imshow(merged_rgb)
plt.title("Merged RGB Image")
plt.axis("off")
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.imshow(h), plt.title("Hue Channel"), plt.axis("off")
plt.subplot(1,3,2), plt.imshow(s), plt.title("Saturation Channel"), plt.axis("off")
plt.subplot(1,3,3), plt.imshow(v), plt.title("Value Channel"), plt.axis("off")
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
combined = np.concatenate((img_rgb, merged_hsv), axis=1)
plt.figure(figsize=(10, 5))
plt.imshow(combined)
plt.title("Original Image  &  Merged HSV Image")
plt.axis("off")
plt.show()
```

## Output:
- **i)** Read and Display an Image.

   1.Read 'Eagle_in_Flight.jpg' as grayscale and display:

 ![image](https://github.com/user-attachments/assets/038ebb08-6f48-40a8-970a-b8a5d25c456c)

 2.Save image as PNG and display:

 ![image](https://github.com/user-attachments/assets/b69c55e4-6a03-41d5-8059-2bd9efbe8bc8)

3.Cropped image:

![image](https://github.com/user-attachments/assets/b6c2d767-67c9-468f-b53c-3e6c90ac4e96)

4.Resize and flip Horizontally:

![image](https://github.com/user-attachments/assets/a8b0b813-cd0d-4a2a-b250-d387341705fa)

5.Read 'Apollo-11-launch.jpg' and Display the final annotated image:

![image](https://github.com/user-attachments/assets/61ed92d3-69c7-431e-b20e-1589134fda68)
 
- **ii)** Adjust Image Brightness.
  
  1.Create brighter and darker images and display:
  
  ![image](https://github.com/user-attachments/assets/cfcf37f8-76fe-4e64-99f9-052e4d6e4c0d)

- **iii)** Modify Image Contrast.

 <img width="1365" height="350" alt="image" src="https://github.com/user-attachments/assets/d1ce6ca4-d234-4b73-91c9-d03e57a3fa68" />


- **iv)** Generate Third Image Using Bitwise Operations.
  
  1.Split 'Boy.jpg' into B, G, R components and display:
<img width="1010" height="281" alt="image" src="https://github.com/user-attachments/assets/330951e5-7628-46dc-bd58-67e6075a0e29" />


2.Merge the R, G, B channels and display:

![image](https://github.com/user-attachments/assets/c84ed5c5-0846-4cda-9901-36e7ba8311d6)

3.Split the image into H, S, V components and display:

<img width="997" height="260" alt="image" src="https://github.com/user-attachments/assets/1ea154c2-147c-4406-b584-f860dc9ae360" />


4.Merge the H, S, V channels and display:

![image](https://github.com/user-attachments/assets/f063b7f8-8e47-4f4a-8568-66f7f29635aa)

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

