import cv2
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk
import time
import os
import platform


# ---------------------------------------------------------------------
# GPIO COMPATIBILITY LAYER
# ---------------------------------------------------------------------

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        LOW = 0
        HIGH = 1

        @staticmethod
        def setmode(mode):
            print(f"[MOCK GPIO] setmode({mode})")

        @staticmethod
        def setup(pin, mode):
            print(f"[MOCK GPIO] setup(pin={pin}, mode={mode})")

        @staticmethod
        def output(pin, value):
            print(f"[MOCK GPIO] output(pin={pin}, value={value})")

        @staticmethod
        def cleanup():
            print("[MOCK GPIO] cleanup()")

    GPIO = MockGPIO()


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

roi_box = None
roi_frame_box = None
detection_roi = None

drawing = False
ix = iy = ex = ey = 0

offset_x = -7
offset_y = 0

ref_captured = False

angle_threshold = 10
motion_threshold = 20
stable_time_required = 0.3

angle_history = []
angle_history_maxlen = 20

frame_width = 800
frame_height = 450

orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

ref_img = None
kp1 = None
des1 = None

current_frame = None
last_roi_gray = None
last_stable_time = 0

capturing = True

BUZZER_PIN = 18
MOTOR_PIN = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(MOTOR_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(MOTOR_PIN, GPIO.LOW)

motor_active_until = 0
flashing_ui = False
flash_end_time = 0


# ---------------------------------------------------------------------
# IMAGE PROCESSING HELPERS
# ---------------------------------------------------------------------

def laplacian_sharpen(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.addWeighted(image, 1.0, lap, -0.3, 0)


def map_roi_to_frame(roi, widget_w, widget_h, frame_w, frame_h):
    x, y, w, h = roi

    scale_x = frame_w / max(widget_w, 1)
    scale_y = frame_h / max(widget_h, 1)

    fx = round(x * scale_x)
    fy = round(y * scale_y)
    fw = round(w * scale_x)
    fh = round(h * scale_y)

    fx = max(0, min(fx, frame_w - 1))
    fy = max(0, min(fy, frame_h - 1))
    fw = max(1, min(fw, frame_w - fx))
    fh = max(1, min(fh, frame_h - fy))

    return fx, fy, fw, fh


def resize_keep_aspect(img, max_size=(200, 200)):
    h, w = img.shape[:2]
    max_w, max_h = max_size

    scale = min(max_w / w, max_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((max_h, max_w, 3), 30, dtype=np.uint8)

    x_offset = (max_w - new_w) // 2
    y_offset = (max_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def resize_with_aspect(image, target_size=(800, 450)):
    img_h, img_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / img_w, target_h / img_h)

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas, x_offset, y_offset, new_w, new_h


# ---------------------------------------------------------------------
# CAMERA OPENING FUNCTION
# ---------------------------------------------------------------------

def open_camera(camera_index=None):
    indexes = [camera_index] if camera_index is not None else [0, 1, 2, 3, 4]

    if platform.system() == "Windows":
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY
        ]
    else:
        backends = [
            cv2.CAP_V4L2,
            cv2.CAP_ANY
        ]

    for backend in backends:
        for index in indexes:
            print(f"[INFO] Trying camera index {index} with backend {backend}")

            cap_obj = cv2.VideoCapture(index, backend)

            if not cap_obj.isOpened():
                cap_obj.release()
                continue

            cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            cap_obj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            ret, frame = cap_obj.read()

            if ret and frame is not None:
                print(f"[INFO] Camera opened successfully on index {index}")
                return cap_obj

            cap_obj.release()

    raise RuntimeError(
        "No working camera found. Check camera connection, camera index, or privacy settings."
    )


# ---------------------------------------------------------------------
# GPIO / OUTPUT CONTROL
# ---------------------------------------------------------------------

def activate_motor(duration):
    global motor_active_until

    GPIO.output(MOTOR_PIN, GPIO.HIGH)
    motor_active_until = time.time() + duration


def trigger_alert(duration=0.3):
    global flashing_ui, flash_end_time

    GPIO.output(BUZZER_PIN, GPIO.HIGH)

    window.configure(bg="#660000")
    main_frame.configure(bg="#660000")
    right_frame.configure(bg="#660000")
    center_container.configure(bg="#660000")
    control_frame.configure(bg="#660000")
    height_slider.configure(bg="#660000", troughcolor="#990000")
    width_slider.configure(bg="#660000", troughcolor="#990000")
    status_label.configure(bg="#660000")
    btn_frame.configure(bg="#660000")

    flashing_ui = True
    flash_end_time = time.time() + duration


def close_app():
    global capturing

    capturing = False

    try:
        if cap.isOpened():
            cap.release()
    except Exception:
        pass

    try:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.output(MOTOR_PIN, GPIO.LOW)
        GPIO.cleanup()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    try:
        if window.winfo_exists():
            window.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------
# GUI CALLBACKS
# ---------------------------------------------------------------------

def styled_btn(parent, text, cmd):
    return tk.Button(
        parent,
        text=text,
        command=cmd,
        width=20,
        font=("JetBrains Mono", 10),
        bg="#282c34",
        fg="white",
        activebackground="#3c3f41",
        activeforeground="#00ffff",
        bd=0,
        relief="flat",
        cursor="hand2"
    )


def show_sliders():
    width_placeholder.grid_forget()
    height_placeholder.grid_forget()

    width_slider.grid(row=1, column=0, padx=10)
    height_slider.grid(row=1, column=1, padx=10)


def update_detection_size(val=None):
    global detection_roi

    if roi_frame_box is None:
        return

    cx = frame_width // 2
    cy = frame_height // 2

    new_w = width_slider.get()
    new_h = height_slider.get()

    dx = cx - (new_w // 2) + offset_x
    dy = cy - (new_h // 2) + offset_y

    dx = max(0, min(dx, frame_width - new_w))
    dy = max(0, min(dy, frame_height - new_h))

    detection_roi = dx, dy, new_w, new_h


def on_mouse(event_type, x, y, flags, param):
    global ix, iy, ex, ey
    global roi_box, drawing, roi_frame_box, detection_roi
    global last_roi_gray, ref_captured

    w = video_label.winfo_width()
    h = video_label.winfo_height()

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    if ref_captured and event_type == cv2.EVENT_LBUTTONDOWN:
        status_var.set("ℹ️ Reset reference to redraw ROI")
        status_label.config(fg="#ffaa00")
        return

    if event_type == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ex, ey = x, y

        status_var.set("✏️ Drawing ROI...")
        status_label.config(fg="#00ffff")

    elif event_type == cv2.EVENT_MOUSEMOVE and drawing:
        ex, ey = x, y

    elif event_type == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        ex, ey = x, y

        x0, y0 = min(ix, ex), min(iy, ey)
        x1, y1 = max(ix, ex), max(iy, ey)

        roi_box = x0, y0, x1 - x0, y1 - y0
        last_roi_gray = None

        status_var.set("🟩 ROI Box Set")
        status_label.config(fg="#00ff66")

        show_sliders()

        if current_frame is not None:
            widget_w = video_label.winfo_width()
            widget_h = video_label.winfo_height()

            frame_h, frame_w = current_frame.shape[:2]

            roi_frame_box = map_roi_to_frame(
                roi_box,
                widget_w,
                widget_h,
                frame_w,
                frame_h
            )

            width = width_slider.get()
            height = height_slider.get()

            cx = frame_width // 2
            cy = frame_height // 2

            dx = cx - (width // 2) + offset_x
            dy = cy - (height // 2) + offset_y

            dx = max(0, min(dx, frame_w - width))
            dy = max(0, min(dy, frame_h - height))

            detection_roi = dx, dy, width, height

        ref_captured = False


def reset_reference():
    global ref_img, kp1, des1
    global roi_box, roi_frame_box, detection_roi
    global ref_captured, angle_history

    ref_img = None
    kp1 = None
    des1 = None

    roi_box = None
    roi_frame_box = None
    detection_roi = None

    ref_captured = False
    angle_history = []

    status_var.set("📝 Draw ROI Box (Reference Reset)")
    status_label.config(fg="#ffaa00")

    width_slider.grid_forget()
    height_slider.grid_forget()

    blank_ref = ImageTk.PhotoImage(
        Image.new("RGB", (200, 200), color="#1e1e1e")
    )

    ref_label.imgtk = blank_ref
    ref_label.configure(image=blank_ref)


def capture_reference():
    global ref_img, kp1, des1
    global last_roi_gray, ref_captured

    if roi_frame_box is None:
        status_var.set("❗ Draw ROI first!")
        status_label.config(fg="#ff4444")
        return

    if current_frame is None:
        status_var.set("❗ No camera frame available!")
        status_label.config(fg="#ff4444")
        return

    fx, fy, fw, fh = roi_frame_box

    roi = current_frame[fy:fy + fh, fx:fx + fw]

    if roi.size == 0:
        status_var.set("❗ Invalid ROI!")
        status_label.config(fg="#ff4444")
        return

    roi = cv2.convertScaleAbs(roi, alpha=1.0, beta=-40)
    sharpened = laplacian_sharpen(roi)

    ref_img = sharpened.copy()

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray, None)

    if des1 is None or len(kp1) < 10:
        status_var.set("⚠️ Not enough reference features. Redraw ROI.")
        status_label.config(fg="#ffaa00")
        ref_img = None
        kp1 = None
        des1 = None
        ref_captured = False
        return

    status_var.set("✅ Reference Captured")
    status_label.config(fg="#00ccff")

    last_roi_gray = None
    ref_captured = True


# ---------------------------------------------------------------------
# MAIN FRAME UPDATE LOOP
# ---------------------------------------------------------------------

def update_frames():
    global current_frame
    global last_roi_gray, last_stable_time
    global angle_history
    global frame_width, frame_height
    global flashing_ui, flash_end_time, motor_active_until

    if time.time() > motor_active_until:
        GPIO.output(MOTOR_PIN, GPIO.LOW)

    if flashing_ui and time.time() > flash_end_time:
        GPIO.output(BUZZER_PIN, GPIO.LOW)

        window.configure(bg=original_bg)
        main_frame.configure(bg=original_bg)
        right_frame.configure(bg=original_bg)
        center_container.configure(bg=original_bg)
        control_frame.configure(bg=original_bg)
        height_slider.configure(bg=original_bg, troughcolor="#44475a")
        width_slider.configure(bg=original_bg, troughcolor="#44475a")
        status_label.configure(bg=original_bg)
        btn_frame.configure(bg=original_bg)

        flashing_ui = False

    ret, frame = cap.read()

    if not window.winfo_exists():
        return

    if not ret or frame is None:
        status_var.set("❌ Camera feed not available")
        status_label.config(fg="#ff4444")
        window.after(100, update_frames)
        return

    frame_height, frame_width = frame.shape[:2]

    frame = cv2.flip(frame, -1)
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=-40)
    frame = laplacian_sharpen(frame)

    current_frame = frame.copy()
    display_frame = frame.copy()

    if drawing:
        x0, y0 = ix, iy
        x1, y1 = ex, ey

        x0 = max(0, min(x0, video_label.winfo_width() - 1))
        y0 = max(0, min(y0, video_label.winfo_height() - 1))
        x1 = max(0, min(x1, video_label.winfo_width() - 1))
        y1 = max(0, min(y1, video_label.winfo_height() - 1))

        fx0, fy0, _, _ = map_roi_to_frame(
            (x0, y0, 1, 1),
            video_label.winfo_width(),
            video_label.winfo_height(),
            frame_width,
            frame_height
        )

        fx1, fy1, _, _ = map_roi_to_frame(
            (x1, y1, 1, 1),
            video_label.winfo_width(),
            video_label.winfo_height(),
            frame_width,
            frame_height
        )

        cv2.rectangle(display_frame, (fx0, fy0), (fx1, fy1), (0, 255, 255), 2)

    elif roi_box and ref_img is None:
        widget_w = video_label.winfo_width()
        widget_h = video_label.winfo_height()

        fx, fy, fw, fh = map_roi_to_frame(
            roi_box,
            widget_w,
            widget_h,
            frame_width,
            frame_height
        )

        cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)

    elif ref_img is not None:
        if ref_captured and roi_frame_box is not None:
            rw, rh = roi_frame_box[2], roi_frame_box[3]

            cx = frame_width // 2
            cy = frame_height // 2

            top_left = (
                cx - rw // 2 + offset_x,
                cy - rh // 2 + offset_y
            )

            bottom_right = (
                cx + rw // 2 + offset_x,
                cy + rh // 2 + offset_y
            )

            cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                "Reference",
                (top_left[0], top_left[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        if detection_roi is not None:
            dx, dy, dw, dh = detection_roi

            cv2.rectangle(
                display_frame,
                (dx, dy),
                (dx + dw, dy + dh),
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Detection",
                (dx, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            roi_color = frame[dy:dy + dh, dx:dx + dw]
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_score = 0

        if last_roi_gray is not None and last_roi_gray.shape != roi_gray.shape:
            last_roi_gray = None

        if last_roi_gray is not None and roi_gray.shape == last_roi_gray.shape:
            diff = cv2.absdiff(roi_gray, last_roi_gray)
            motion_score = np.mean(diff)

            if motion_score > motion_threshold:
                last_stable_time = time.time()
        else:
            last_stable_time = time.time()

        last_roi_gray = roi_gray.copy()

        roi_brightness = np.mean(roi_gray)
        pcb_present = roi_brightness < 200

        if not pcb_present:
            status_var.set("🕐 Waiting for PCB...")
            status_label.config(fg="#888888")

        elif motion_score > motion_threshold:
            status_var.set("🚚 PCB Moving...")
            status_label.config(fg="#ffaa00")

        elif time.time() - last_stable_time > stable_time_required:
            blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            _, thresh = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            fully_inside = False

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                fill_ratio = area / (roi_gray.shape[0] * roi_gray.shape[1])

                _, _, w, h = cv2.boundingRect(largest)

                min_fill_ratio = 0.7
                min_width = roi_gray.shape[1] * 0.5
                min_height = roi_gray.shape[0] * 0.5

                fully_inside = (
                    fill_ratio > min_fill_ratio and
                    w > min_width and
                    h > min_height
                )

            if not fully_inside:
                status_var.set("🟨 Place PCB Fully Inside the Frame")
                status_label.config(fg="#ffff33")

            else:
                kp2, des2 = orb.detectAndCompute(roi_gray, None)

                if des1 is None or des2 is None:
                    matches = []
                else:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) >= 10:
                    good_matches = matches[:50]

                    src_pts = np.float32(
                        [kp1[m.queryIdx].pt for m in good_matches]
                    ).reshape(-1, 1, 2)

                    dst_pts = np.float32(
                        [kp2[m.trainIdx].pt for m in good_matches]
                    ).reshape(-1, 1, 2)

                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if H is not None:
                        angle_rad = math.atan2(H[1, 0], H[0, 0])
                        angle_deg = (math.degrees(angle_rad) + 360) % 360
                        adjusted = min(angle_deg, 360 - angle_deg)

                        angle_history.append(adjusted)

                        if len(angle_history) > angle_history_maxlen:
                            angle_history.pop(0)

                        smoothed_angle = sum(angle_history) / len(angle_history)

                        if adjusted < angle_threshold:
                            status_var.set(f"📐 {smoothed_angle:.1f}° — ✅ OK")
                            status_label.config(fg="#00ff66")
                            activate_motor(duration=5.0)
                        else:
                            status_var.set(f"📐 {smoothed_angle:.1f}° — ❌ Wrong")
                            status_label.config(fg="#ff4444")
                            trigger_alert()
                    else:
                        status_var.set("🚫 Can't Detect")
                        status_label.config(fg="#ff6666")
                else:
                    status_var.set("🚫 Can't Detect")
                    status_label.config(fg="#ff6666")

    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    resized_rgb, _, _, _, _ = resize_with_aspect(frame_rgb, (800, 450))

    imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized_rgb))

    if window.winfo_exists():
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if ref_img is not None:
            ref_thumb = resize_keep_aspect(ref_img, (200, 200))
            ref_rgb = cv2.cvtColor(ref_thumb, cv2.COLOR_BGR2RGB)
            ref_imgtk = ImageTk.PhotoImage(image=Image.fromarray(ref_rgb))

            ref_label.imgtk = ref_imgtk
            ref_label.configure(image=ref_imgtk)

        window.after(10, update_frames)


# ---------------------------------------------------------------------
# GUI SETUP
# ---------------------------------------------------------------------

window = tk.Tk()
window.title("PCB Orientation Detector")
window.configure(bg="#000000")

FULLSCREEN_MODE = platform.system() != "Windows"
window.attributes("-fullscreen", FULLSCREEN_MODE)

original_bg = "#000000"

window.lift()
window.attributes("-topmost", True)
window.focus_force()


def keep_window_on_top():
    try:
        window.lift()
        window.attributes("-topmost", True)
        window.focus_force()
    except Exception:
        pass

    window.after(2000, keep_window_on_top)


keep_window_on_top()


def toggle_fullscreen(event=None):
    current = window.attributes("-fullscreen")
    window.attributes("-fullscreen", not current)


window.bind("<f>", toggle_fullscreen)
window.protocol("WM_DELETE_WINDOW", close_app)

icon_path = os.path.abspath("cpu.png")

if os.path.exists(icon_path):
    try:
        pil_img = Image.open(icon_path)
        img = ImageTk.PhotoImage(pil_img)
        window.iconphoto(True, img)
    except Exception as e:
        print("Icon set error:", e)


main_frame = tk.Frame(window, bg="#000000")
main_frame.pack(padx=10, pady=10, fill="both", expand=True)

video_label = tk.Label(main_frame, bg="#1e1e1e", bd=3, relief="ridge")
video_label.grid(row=0, column=0, rowspan=2, padx=0, pady=0)

main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(1, weight=1)

video_label.bind(
    "<Button-1>",
    lambda e: on_mouse(cv2.EVENT_LBUTTONDOWN, e.x, e.y, None, None)
)

video_label.bind(
    "<B1-Motion>",
    lambda e: on_mouse(cv2.EVENT_MOUSEMOVE, e.x, e.y, None, None)
)

video_label.bind(
    "<ButtonRelease-1>",
    lambda e: on_mouse(cv2.EVENT_LBUTTONUP, e.x, e.y, None, None)
)

right_frame = tk.Frame(main_frame, bg="#000000")
right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(10, 0))

right_frame.rowconfigure(0, weight=1)
right_frame.rowconfigure(1, weight=0)
right_frame.rowconfigure(2, weight=1)
right_frame.columnconfigure(0, weight=1)

center_container = tk.Frame(right_frame, bg="#000000")
center_container.grid(row=1, column=0, sticky="nsew")

ref_frame = tk.Frame(center_container, bg="#1e1e1e", bd=2, relief="ridge")
ref_frame.pack(pady=10)

ref_label = tk.Label(ref_frame, bg="#1e1e1e")
ref_label.pack()

blank_ref = ImageTk.PhotoImage(
    Image.new("RGB", (200, 200), color="#1e1e1e")
)

ref_label.imgtk = blank_ref
ref_label.configure(image=blank_ref)

btn_frame = tk.Frame(center_container, bg="#000000")
btn_frame.pack(pady=10)

btn_capture = styled_btn(btn_frame, "📸 Capture Ref", capture_reference)
btn_capture.pack(pady=8)

btn_reset = styled_btn(btn_frame, "🔄 Reset Ref", reset_reference)
btn_reset.pack(pady=8)

btn_exit = styled_btn(btn_frame, "❌ Exit", close_app)
btn_exit.pack(pady=8)

control_frame = tk.Frame(main_frame, bg="#000000")
control_frame.grid(row=3, column=0, pady=(10, 10))

width_placeholder = tk.Frame(control_frame, width=375, height=53, bg="#000000")
height_placeholder = tk.Frame(control_frame, width=375, height=53, bg="#000000")

width_placeholder.grid(row=1, column=0, padx=10, pady=(1, 1))
height_placeholder.grid(row=1, column=1, padx=10, pady=(1, 1))

width_slider = tk.Scale(
    control_frame,
    from_=50,
    to=frame_width,
    orient="horizontal",
    label="                               Detection Width",
    bg="#000000",
    fg="white",
    troughcolor="#44475a",
    sliderlength=20,
    length=375,
    highlightthickness=0,
    bd=0,
    relief="flat",
    activebackground="#50fa7b",
    command=update_detection_size,
    font=("JetBrains Mono", 10)
)

width_slider.set(800)

height_slider = tk.Scale(
    control_frame,
    from_=50,
    to=frame_height,
    orient="horizontal",
    label="                               Detection Height",
    bg="#000000",
    fg="white",
    troughcolor="#44475a",
    sliderlength=20,
    length=375,
    highlightthickness=0,
    bd=0,
    relief="flat",
    activebackground="#50fa7b",
    command=update_detection_size,
    font=("JetBrains Mono", 10)
)

height_slider.set(450)

status_var = tk.StringVar(value="📝 Draw ROI Box")

status_label = tk.Label(
    main_frame,
    textvariable=status_var,
    font=("JetBrains Mono", 11, "bold"),
    bg="#000000",
    fg="light gray",
    wraplength=800,
    justify="center",
    anchor="center",
    height=1
)

status_label.grid(row=2, column=0, pady=(10, 5), sticky="ew")


# ---------------------------------------------------------------------
# CAMERA INIT + MAIN LOOP
# ---------------------------------------------------------------------

cap = open_camera()
last_stable_time = time.time()

window.after(0, update_frames)
window.mainloop()