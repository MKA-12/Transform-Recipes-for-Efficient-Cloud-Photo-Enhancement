import image_dehazer
import cv2
HazeImg = cv2.imread('../transformed/upscaledInput.jpg')					
HazeCorrectedImg = image_dehazer.remove_haze(HazeImg)
# print(HazeCorrectedImg.shape)
HazeOriginal = cv2.imread('../inputs/hazed.jpg')
HazeOriginal = image_dehazer.remove_haze(HazeOriginal) 
cv2.imwrite('../transformed/out.jpg',HazeCorrectedImg)
cv2.imwrite('../transformed/outO.jpg',HazeOriginal)