import cv2
import numpy as np

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, refine the corners and add object & image points
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print("Captured (", len(imgpoints), ") Images for Calibration. Press C to calibrate.")

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (7, 6), corners2, ret)

    # Display the frame
    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1)

    # Press 'q' to break out of the loop
    if key == ord('q'):
        break

    # Press 'c' to calibrate and display results
    if key == ord('c') and len(imgpoints) > 5:  # Use at least 5 captures for calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("\nCalibration results:")
        print("\nCamera matrix (mtx): \n", mtx, "\n")
        print("\nDistortion coefficients (dist): \n", dist, "\n")
        break

cap.release()
cv2.destroyAllWindows()