from math import log10, sqrt 
import cv2 
import numpy as np 

def PSNR(original, compressed): 
	mse = np.mean((original - compressed) ** 2) 
	if(mse == 0): # MSE is zero means no noise is present in the signal . 
				# Therefore PSNR have no importance. 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr 

def main(): 
	original = cv2.imread("../images/upsized.jpg")
	# readImg = np.vstack([original,np.zeros((1,original.shape[1],3),dtype = 'uint8')])
	# readImg = np.hstack([readImg,np.zeros((readImg.shape[0],4,3),dtype = 'uint8')])

	# print(original.shape)

	compressed = cv2.imread("../images/clientO.jpg")
	# original = cv2.resize(original,(compressed.shape[1],compressed.shape[0])) 
	# compressed = cv2.resize(compressed,(original.shape[1],original.shape[0])) 
	# compressed = compressed[:-1,:-4,:] 
	value = PSNR(original, compressed) 
	print(f"PSNR value is {value} dB") 
	
if __name__ == "__main__": 
	main() 
