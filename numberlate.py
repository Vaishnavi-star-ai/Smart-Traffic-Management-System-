import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt

def detect_license_plate(img_path):
    # 1. Load Image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not loaded")
        return
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # ✅ Edge detection added here

    # 3. Find Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # 4. Detect License Plate
    plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # Rectangular shape
            plate_contour = approx
            break

    if plate_contour is None:
        print("No license plate found")
        return

    # 5. Extract Plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    cv2.drawContours(img, [plate_contour], 0, (0, 255, 0), 3)
    
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_plate = gray[topx:bottomx+1, topy:bottomy+1]

    # 6. OCR Processing
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_plate)

    # 7. Display Results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(4):
        pt1 = plate_contour[i][0]
        pt2 = plate_contour[(i + 1) % 4][0]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-')
    plt.title('Detected Plate Location')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cropped_plate, cmap='gray')
    
    if result:
        text = result[0][1]
        confidence = result[0][2]
        plt.title(f'Extracted: {text}\nConfidence: {confidence:.2f}')
    else:
        plt.title('No text detected')
    
    plt.tight_layout()
    plt.show()

# Usage
detect_license_plate('models/image_123.jpg')
