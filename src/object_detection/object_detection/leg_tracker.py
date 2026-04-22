import cv2
from ultralytics import YOLO

# Lower-body landmark indices (COCO keypoint format)
LOWER_BODY_KEYPOINTS = {
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

def get_follow_target(keypoints):
    """Calculate the midpoint between ankles as the follow target."""
    left_ankle  = keypoints[15]
    right_ankle = keypoints[16]

    # Only use if both ankles are visible (confidence > 0.5)
    if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
        mid_x = int((left_ankle[0] + right_ankle[0]) / 2)
        mid_y = int((left_ankle[1] + right_ankle[1]) / 2)
        return (mid_x, mid_y), [left_ankle, right_ankle]
    return None

def draw_lower_body(frame, keypoints):
    """Draw only the lower body keypoints and skeleton."""
    # Draw keypoints
    for idx, name in LOWER_BODY_KEYPOINTS.items():
        kp = keypoints[idx]
        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
        if conf > 0.5:
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(frame, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Draw skeleton lines: hip→knee→ankle
    connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
    for a, b in connections:
        kp_a, kp_b = keypoints[a], keypoints[b]
        if kp_a[2] > 0.5 and kp_b[2] > 0.5:
            cv2.line(frame,
                     (int(kp_a[0]), int(kp_a[1])),
                     (int(kp_b[0]), int(kp_b[1])),
                     (0, 200, 255), 2)

def main():
    model = YOLO("yolov8n-pose.pt")  # downloads automatically on first run
    cap   = cv2.VideoCapture(0)      # 0 = webcam, or replace with video path

    print("Press 'q' to quit")
    image = cv2.imread("C:/Users/Applesauce/Pictures/Camera Roll/WIN_20260420_17_23_37_Pro.jpg")

    #while cap.isOpened():
        #ret, frame = cap.read()
        #if not ret:
        #    break

    results = model(image, verbose=False)
    targets = []
    for result in results:
            if result.keypoints is None:
                continue

            for person_kps in result.keypoints.data:
                keypoints = person_kps.cpu().numpy()  # shape: (17, 3) → x, y, conf

                draw_lower_body(image, keypoints)

                # Get the follow target (midpoint of ankles)
                target, ankles = get_follow_target(keypoints)
                targets.append([target,ankles])
                if target:
                    cv2.circle(image, target, 10, (0, 0, 255), -1)
                    cv2.putText(image, "TARGET", (target[0] + 12, target[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    print(targets)
    cv2.imshow("Leg Tracker", image)
    cv2.waitKey(0) 

if __name__ == "__main__":
    main()