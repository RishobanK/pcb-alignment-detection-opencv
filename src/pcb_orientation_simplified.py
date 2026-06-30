import cv2 as cv
import numpy as np
import time
import threading

RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

# Camera Config
IP = "192.168.42.129"
URL = f"http://{IP}:8080/video"

class LatestFrameCamera:
    def __init__(self, url):
        self.cap = cv.VideoCapture(url)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def isOpened(self):
        return self.cap.isOpened()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# cap = cv.VideoCapture(0)            # Default Cam
# cap = cv.VideoCapture(URL)          # Android Cam
cap = LatestFrameCamera(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

def select_roi(frameResize):
    roi_box = cv.selectROI("frame", frameResize, False)
    
    return roi_box

def extract_features(image, orb):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    keypoint_image = cv.drawKeypoints(image, keypoints, None)
    
    return keypoints, descriptors, keypoint_image

def match_features(ref_des, live_des, bf):
    if ref_des is None or live_des is None:
        return[]
    
    matches = bf.match(ref_des, live_des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def filter_matches(matches, distance_threshold):
    good_matches = []
    
    for match in matches:
        if match.distance < distance_threshold:
            good_matches.append(match)
            
    return good_matches

def homography(ref_kp, live_kp, good_matches):
    if len(good_matches) < 10:
        return None, None
    
    ref_points = []
    live_points = []
    
    for match in good_matches:
        ref_points.append(ref_kp[match.queryIdx].pt)
        live_points.append(live_kp[match.trainIdx].pt)
        
    ref_points = np.float32(ref_points).reshape(-1, 1, 2)
    live_points = np.float32(live_points).reshape(-1, 1, 2)
    
    H, mask = cv.findHomography(ref_points, live_points, cv.RANSAC, 5.0)
    
    return H, mask

def reset_detection():
    roi_selected = False
    roi_box = None
    ref_captured = False
    ref_img = None

    ref_kp = None
    ref_des = None
    ref_kp_img = None

    angle_text = "Angle: --"
    status_text = "Status: No reference"
    status_color = (0, 255, 255)

    homography_text = "Homography: --"
    homography_color = (255, 255, 255)

    return (roi_selected, roi_box, ref_captured, ref_img, ref_kp, ref_des, ref_kp_img, angle_text, status_text, status_color, homography_text, homography_color)
    
def main():
    previous_time = time.perf_counter()         # FPS
    detection_time = 0
    
    roi_selected = False                        # ROI
    roi_box = None
    
    ref_captured = False                        # Reference image
    ref_img = None
    roi = None
    
    ref_kp = None                               # Ref Keypoints
    ref_des = None                              # Ref Descriptors
    ref_kp_img = None                   
    
    live_kp = None                              # Live Keypoints
    live_des = None                             # Live Descriptors
    live_kp_img = None
    
    matches = []                                # BFMatcher
    match_count = 0
    match_image = None
    
    good_matches = []                           # Filter matches
    good_match_count = 0
    good_match_distance = 40
    good_match_img = None
    
    H = None                                    # Homography
    homography_mask = None
    homography_found = False
    
    angle_tolerance = 10
    
    min_inliers = 10
    min_inlier_ratio = 0.20
    inlier_ratio = 0
    
    angle_text = "Angle: --"
    status_text = "Press R to select reference"
    status_color = (0, 255, 255)
    
    raw_status = "WAITING"
    raw_status_color = (0, 255, 255)
    
    stable_status = "Press R to select reference"
    last_raw_status = None
    same_status_count = 0
    req_status_frames = 3
    
    homography_text = "Homography: --"
    homography_color = (0, 255, 255)
    
    inlier_count = 0
    live_kp_count = 0
    
    orb = cv.ORB_create(nfeatures = 1000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Waiting for first frame...")
            time.sleep(0.05)
            continue
        
        frameResize = cv.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))        
        displayFrame = frameResize.copy()
        
        # FPS Calculation
        current_time = time.perf_counter()
        frame_time = current_time - previous_time
        fps = 1 / frame_time
        previous_time = current_time
        
        height, width, channels = frameResize.shape
        
        resolution_text = f"Resolution: {width}x{height}"
        fps_text = f"FPS: {fps:.1f}"
        
        # cv.putText(displayFrame, resolution_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv.putText(displayFrame, fps_text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # ROI
        if roi_selected:
            x, y, w, h = roi_box
            
            cv.rectangle(displayFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi = frameResize[y:y + h, x:x +w]
            live_img = frameResize
            if ref_captured:
                detection_start_time = time.perf_counter()
                
                live_kp, live_des, live_kp_img = extract_features(live_img, orb)
                matches = match_features(ref_des, live_des, bf)
                match_count = len(matches)
                
                good_matches = filter_matches(matches, good_match_distance)
                good_match_count = len(good_matches)
                
                H, homography_mask = homography(ref_kp, live_kp, good_matches)
                
                if live_des is not None:
                    live_kp_count = len(live_kp)
                else:
                    live_kp_count = 0
                
            #     if match_count > 0:
            #         match_image = cv.drawMatches(ref_img, ref_kp, live_img, live_kp, matches[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #     else:
            #         match_image = None
                
            #     if good_match_count > 0:
            #         good_match_img = cv.drawMatches(ref_img, ref_kp, live_img, live_kp, good_matches[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # #       bad_match_image = cv.drawMatches(ref_img, ref_kp, live_img, live_kp, good_matches[-30:], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # #       cv.imshow("Bad Matches", bad_match_image)
            #     else:
            #         good_match_img = None
                    
                if homography_mask is not None:
                    inlier_count = int(homography_mask.sum())
                else:
                    inlier_count = 0
                    
                if good_match_count > 0:
                    inlier_ratio = inlier_count / good_match_count
                else:
                    inlier_ratio = 0
                    
                if H is not None and inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio:
                    # homography_found = True
                    ref_h, ref_w = ref_img.shape[:2]
                    
                    ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
                    detected_corners = cv.perspectiveTransform(ref_corners, H)
                    detected_corners = np.int32(detected_corners)
                    
                    cv.polylines(displayFrame, [detected_corners], True, (0, 255, 0), 2)
                    
                    pt1 = detected_corners[0][0]
                    pt2 = detected_corners[1][0]
                    
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    
                    angle = np.degrees(np.arctan2(dy, dx))
                    if abs(angle) <= angle_tolerance:
                        raw_status = "OK"
                        raw_status_color = (0, 255, 0)
                    else:
                        raw_status = "Adjust PCB"
                        raw_status_color = (0, 0, 255)
                    
                    # cv.putText(displayFrame, f"angle: {angle:.1f} deg", (20, 210), cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                    # cv.putText(displayFrame, f"Status: {status_text}", (20, 240), cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                    # cv.putText(displayFrame, f"inliers: {inlier_count}", (20, 270), cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                    
                    angle_text = f"Angle: {angle:.1f} deg"
                        
                    homography_text = "Homography: FOUND"
                    homography_color = (0, 255, 0)
                else:
                    # homography_found = False
                    # cv.putText(displayFrame, f"angle: --", (20, 210), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv.putText(displayFrame, f"Status: LOW Confidence", (20, 240), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv.putText(displayFrame, f"inliers: {inlier_count}", (20, 270), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                   # angle_text = "Angle: --"
                   # status_text = "Status: LOW Confidence"
                    status_color = (0, 0, 255)
                        
                    homography_text = "Homography: LOW"
                    homography_color = (0, 0, 255)
                    
                    raw_status = "LOW CONFIDENCE"
                    raw_status_color = (0, 0, 255)
                    
                if raw_status == last_raw_status:
                    same_status_count += 1
                else:
                    same_status_count = 1
                    last_raw_status = raw_status
                    
                if same_status_count >= req_status_frames:
                    stable_status = raw_status
                    status_color = raw_status_color
                    
                status_text = f"Status: {stable_status}"
                
                detection_end_time = time.perf_counter()
                detection_time = (detection_end_time - detection_start_time) * 1000
                
                # if homography_found:
                #     cv.putText(displayFrame, "Homography: FOUND", (20, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # else:
                #     cv.putText(displayFrame, "Homography: LOW", (20, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                # cv.putText(displayFrame, f"Live keypoints: {live_kp_count}", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # cv.putText(displayFrame, f"Matches: {match_count}", (20, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # cv.putText(displayFrame, f"Good matches: {good_match_count}", (20, 150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # cv.putText(displayFrame, f"Detection time: {detection_time:.1f} ms", (20, 300), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
        # if ref_captured:
        #     cv.imshow("Reference", ref_img)
            
        #     if ref_kp_img is not None:
        #         cv.imshow("Reference Keypoints", ref_kp_img)
        #     if live_kp_img is not None:
        #         cv.imshow("Live Keypoints", live_kp_img)
        #     if match_image is not None:
        #         cv.imshow("Matches", match_image)
        #     if good_match_img is not None:
        #         cv.imshow("Good Matches", good_match_img)
        
        bg_x = 0
        bg_y = 0
        bg_w = 280
        bg_h = 275
        
        cv.rectangle(displayFrame, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (25, 25, 25), -1)
        cv.rectangle(displayFrame, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (120, 120, 120), 1)
        
        cv.putText(displayFrame, "        PCB ALIGNMENT", (10, 20),
           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.putText(displayFrame, resolution_text, (20, 45),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv.putText(displayFrame, fps_text, (20, 65),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv.putText(displayFrame, f"Live keypoints: {live_kp_count}", (20, 85),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv.putText(displayFrame, f"Matches: {match_count}", (20, 105),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv.putText(displayFrame, f"Good matches: {good_match_count}", (20, 125),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv.putText(displayFrame, homography_text, (20, 145),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, homography_color, 1)

        cv.putText(displayFrame, angle_text, (20, 165),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        cv.putText(displayFrame, f"Inliers: {inlier_count}", (20, 185),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        cv.putText(displayFrame, f"Inlier ratio: {inlier_ratio:.2f}", (20, 205),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        cv.putText(displayFrame, f"Detection: {detection_time:.1f} ms", (20, 225),
                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv.putText(displayFrame, status_text, (20, 255),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        
        cv.imshow("frame", displayFrame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_box = select_roi(frameResize)
            x, y, w, h = selected_box
            
            if w > 0 and h > 0:
                roi_box = selected_box
                roi_selected = True
                
                ref_img = frameResize[y:y + h, x:x +w].copy()
                ref_captured = True
                
                ref_kp, ref_des, ref_kp_img = extract_features(ref_img, orb)
                
                angle_text = "Angle: --"
                status_text = "Status: Reference captured"
                status_color = (0, 255, 255)

                homography_text = "Homography: --"
                homography_color = (255, 255, 255)

                same_status_count = 0
                last_raw_status = None
                
            #     print("ROI & Ref captured")
            #     if ref_des is None:
            #         print("No ref_des found")
            #     else:
            #         print("Ref keypoints: ", len(ref_kp))
            # else:
            #     print("ROI not selected")
        elif key == ord('c'):
            (roi_selected, roi_box, ref_captured, ref_img, ref_kp, ref_des, ref_kp_img, angle_text, status_text, status_color, homography_text, homography_color) = reset_detection()

            # print("Reference cleared")
                 
    cap.release()
    cv.destroyAllWindows()

main()
