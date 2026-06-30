# Approach Notes: Simplified PCB Alignment Detection Demo

## Purpose

This version was created as a simplified and more explainable PCB alignment detection demo. The earlier version included a GUI, GPIO output, motion/stability checks, and additional image-processing logic. This simplified version focuses mainly on the computer vision pipeline used to detect the PCB orientation.

The goal is to check whether a PCB is aligned correctly compared to a captured reference image.

## Main Pipeline

The system follows this flow:

1. Capture live camera frame
2. Select a reference PCB region of interest
3. Extract ORB features from the reference image
4. Extract ORB features from the live frame
5. Match features using BFMatcher with Hamming distance
6. Filter weak matches
7. Estimate homography using RANSAC
8. Transform reference corners into the live frame
9. Calculate PCB angle using the detected top edge
10. Display OK, ADJUST PCB, or LOW CONFIDENCE status

## Why ORB Was Used

ORB was selected because it is a lightweight feature detection and description method. It is faster than heavier methods such as SIFT or SURF, which makes it more suitable for Raspberry Pi-based real-time testing.

ORB detects useful visual features such as PCB holes, pad corners, silkscreen markings, component edges, and track edges. These features can be matched between the selected reference PCB and the live camera frame.

## Feature Matching

After ORB detects keypoints and descriptors, BFMatcher is used to compare the reference descriptors with the live-frame descriptors.

Since ORB produces binary descriptors, Hamming distance is used for matching. A lower Hamming distance means two descriptors are more similar.

Cross-check matching is used to reduce false matches. This means a match is accepted only when both descriptors mutually select each other as the best match.

## Good Match Filtering

Not every match from the matcher is reliable. Therefore, matches are filtered using descriptor distance. Matches with lower distance are kept as good matches, while weaker matches are rejected.

This improves the quality of the points passed to the homography stage.

## Homography and RANSAC

Homography is used to estimate the geometric transformation between the reference PCB image and the live camera frame.

It can handle movement, rotation, scale change, and slight perspective changes.

RANSAC is used inside homography estimation to reject incorrect feature matches. Correct matches that agree with the final homography are called inliers. Wrong matches are treated as outliers.

The system checks the inlier count before trusting the homography result.

## Inlier Ratio Confidence

In addition to inlier count, an inlier ratio check was added.

Inlier ratio is calculated as:

inlier ratio = inlier count / good match count

This gives a better confidence measure than inlier count alone. For example, 10 inliers out of 12 good matches is strong, but 10 inliers out of 100 good matches is weaker.

This helps avoid trusting weak homography results.

## Angle Calculation

After a reliable homography is found, the four reference PCB corners are transformed into the live frame using perspective transform.

The top-left and top-right detected corners are used to calculate the PCB angle.

The angle is calculated using atan2 because it gives a signed angle between -180 degrees and +180 degrees.

If the absolute angle is within the allowed tolerance, the PCB is considered OK. Otherwise, the system displays ADJUST PCB.

## Temporal Smoothing

Temporal smoothing was added to reduce flickering in the displayed status.

Without smoothing, the status may rapidly switch between OK, ADJUST PCB, and LOW CONFIDENCE due to one-frame detection errors.

With smoothing, the displayed status changes only after the same result appears for a few continuous frames. This makes the demo more stable and easier to observe.

## Camera Latency Consideration

When using an Android phone/IP camera, latency can occur because frames are encoded by the phone, transmitted over WiFi, decoded by Raspberry Pi, and then processed by OpenCV.

After reference capture, the Raspberry Pi also performs ORB matching and homography processing. If frames are processed slower than they arrive, old frames may queue up and the displayed feed can look delayed.

A latest-frame camera approach can be used later to reduce this issue. In that approach, a separate thread continuously reads camera frames and keeps only the newest frame.

## Current Status Outputs

The system displays:

* OK: PCB is detected and angle is within tolerance
* ADJUST PCB: PCB is detected but angle is outside tolerance
* LOW CONFIDENCE: Homography is not reliable enough
* Angle: calculated PCB orientation angle
* Good matches: number of filtered matches
* Inliers: number of geometrically consistent matches
* Inlier ratio: confidence ratio
* Detection time: time taken for the detection pipeline

## Limitations

The system can still struggle under:

* strong glare
* very blurry camera frames
* poor lighting
* very plain PCB areas with few features
* large perspective distortion
* PCB partially outside the camera view
* using a different PCB design from the selected reference

## Future Improvements

Possible future improvements include:

* latest-frame threaded camera capture
* Pi Camera v2 support using Picamera2
* X/Y position offset checking
* optional glare masking
* better GUI layout
* detection only every 2nd or 3rd frame for smoother Raspberry Pi performance
* saving test results and screenshots automatically

## Summary

This simplified version focuses on a clean and explainable OpenCV pipeline. It uses ORB for feature extraction, BFMatcher with Hamming distance for matching, homography with RANSAC for geometric verification, inlier confidence for reliability checking, and atan2-based angle calculation for PCB alignment decision.
