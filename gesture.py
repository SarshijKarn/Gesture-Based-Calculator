import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import math

# Initialize Text-to-Speech Engine
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing pyttsx3: {e}")
    engine = None

def speak(text):
    """Function to speak the given text if the engine is available."""
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error during speech: {e}")
    else:
        print(f"Speech output: {text}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# Application States
STATE_GET_FIRST_NUMBER = "GET_FIRST_NUMBER"
STATE_GET_OPERATOR = "GET_OPERATOR"
STATE_GET_SECOND_NUMBER = "GET_SECOND_NUMBER"
STATE_SHOW_RESULT = "SHOW_RESULT"

# Gesture Stabilization
GESTURE_CONFIRMATION_FRAMES = 10  # Number of consecutive frames to confirm a gesture

# UI Configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG_COLOR = (20, 20, 20)
BG_ALPHA = 0.7
TEXT_COLOR = (245, 245, 245)
HIGHLIGHT_COLOR = (94, 215, 255)  # Light Blue
RESULT_COLOR = (80, 255, 120)    # Greenish
ERROR_COLOR = (0, 80, 255)       # Orange/Red
PROGRESS_BG = (35, 35, 35)
PROGRESS_COLOR = (255, 175, 0)
OPERATOR_COLORS = {"+": (0,255,255), "-": (195,75,185), "*": (80,120,255), "/": (255,110,80)}

# --- Core Logic Functions ---

def get_finger_count(hand_landmarks, hand_label):
    if hand_landmarks is None:
        return 0

    # Landmark indices for fingertips
    tip_ids = [4, 8, 12, 16, 20]
    landmarks = hand_landmarks.landmark
    finger_count = 0

    # --- Thumb ---
    thumb_tip_x = landmarks[tip_ids[0]].x
    thumb_ip_x = landmarks[tip_ids[0] - 2].x

    if hand_label.lower() == 'right':
        if thumb_tip_x < thumb_ip_x:
            finger_count += 1
    else: # Left hand
        if thumb_tip_x > thumb_ip_x:
            finger_count += 1

    # --- Other Four Fingers ---
    for i in range(1, 5):
        tip_y = landmarks[tip_ids[i]].y
        pip_y = landmarks[tip_ids[i] - 2].y
        if tip_y < pip_y:
            finger_count += 1
    return finger_count

def get_operator_gesture(hand_landmarks):
    if hand_landmarks is None:
        return None

    landmarks = hand_landmarks.landmark
    tip_ids = [4, 8, 12, 16, 20]

    # Check if thumb is up
    thumb_tip_y = landmarks[tip_ids[0]].y
    thumb_mcp_y = landmarks[tip_ids[0] - 2].y
    is_thumb_up = thumb_tip_y < thumb_mcp_y

    if not is_thumb_up:
        return None

    # Check which other finger is up
    fingers_up = [landmarks[tip_id].y < landmarks[tip_id - 2].y for tip_id in tip_ids[1:]]

    # Thumb + Index -> Addition
    if fingers_up == [True, False, False, False]:
        return '+'
    # Thumb + Middle -> Subtraction
    elif fingers_up == [False, True, False, False]:
        return '-'
    # Thumb + Ring -> Multiplication
    elif fingers_up == [False, False, True, False]:
        return '*'
    # Thumb + Little -> Division
    elif fingers_up == [False, False, False, True]:
        return '/'
    return None

def draw_rounded_rectangle(img, pt1, pt2, color, radius=15, thickness=-1, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=thickness)
    for corner in [(pt1[0]+radius, pt1[1]+radius), (pt2[0]-radius, pt1[1]+radius),
                   (pt1[0]+radius, pt2[1]-radius), (pt2[0]-radius, pt2[1]-radius)]:
        cv2.circle(overlay, corner, radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img

def draw_ui(frame, prompt_text, current_equation, result_text, error_text,
            gesture_progress, op_symbol=None):
    h, w, _ = frame.shape
    # Semi-transparent rounded background for prompt
    draw_rounded_rectangle(frame, (20, 10), (w-20, 65), BG_COLOR, 18, -1, BG_ALPHA)
    cv2.putText(frame, prompt_text, (40, 50), FONT, 1.05, HIGHLIGHT_COLOR, 2, cv2.LINE_AA)

    # Progress bar for gesture confirmation
    if gesture_progress is not None and gesture_progress > 0:
        max_bar_w = 330
        px = int(max_bar_w * gesture_progress)
        bar_y = 75
        draw_rounded_rectangle(frame, (40, bar_y), (40+max_bar_w, bar_y+22), PROGRESS_BG, 10, -1, 0.9)
        draw_rounded_rectangle(frame, (40, bar_y), (40+px, bar_y+22), PROGRESS_COLOR, 10, -1, 0.8)

    # Operator symbol in color, if applicable
    if op_symbol and op_symbol in OPERATOR_COLORS:
        col = OPERATOR_COLORS[op_symbol]
        cv2.putText(frame, op_symbol, (int(w/2)-30, 170), FONT, 3.7, col, 9, cv2.LINE_AA)

    # Current equation line (rounded bg)
    eqn_bg_top = h - 88
    draw_rounded_rectangle(frame, (20, eqn_bg_top), (w-20, h-25), BG_COLOR, 18, -1, BG_ALPHA)
    cv2.putText(frame, current_equation, (40, h-40), FONT, 1.65, TEXT_COLOR, 3, cv2.LINE_AA)

    # Result centered large, in green
    if result_text:
        textsize = cv2.getTextSize(result_text, FONT, 2.5, 6)[0]
        tx = int((w - textsize[0]) / 2)
        ty = int(h/2) + 10
        draw_rounded_rectangle(frame, (tx-35, ty-textsize[1]-26), (tx+textsize[0]+35, ty+26), (32,90,35), 32, -1, 0.84)
        cv2.putText(frame, result_text, (tx, ty), FONT, 2.5, RESULT_COLOR, 6, cv2.LINE_AA)

    # Error messages
    if error_text:
        textsize = cv2.getTextSize(error_text, FONT, 1.25, 3)[0]
        tx = int((w - textsize[0]) / 2)
        ty = int(h/2) + 88
        draw_rounded_rectangle(frame, (tx-16, ty-textsize[1]-16), (tx+textsize[0]+16, ty+10), ERROR_COLOR, 14, -1, 0.68)
        cv2.putText(frame, error_text, (tx, ty), FONT, 1.25, (255,255,255), 2, cv2.LINE_AA)
    return frame

# --- Main Application Loop ---
def main():
    # Application state variables
    current_state = STATE_GET_FIRST_NUMBER
    first_number = None
    second_number = None
    operator = None
    result = None
    
    # Gesture stabilization variables
    gesture_counter = 0
    last_detected_value = None
    
    # Result display timer
    result_display_start_time = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(rgb_frame)

        # --- Gesture Detection ---
        detected_value = None
        error_message = ""

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_state in [STATE_GET_FIRST_NUMBER, STATE_GET_SECOND_NUMBER]:
                total_fingers = 0
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[i].classification[0].label
                    total_fingers += get_finger_count(hand_landmarks, hand_label)
                detected_value = total_fingers
            
            elif current_state == STATE_GET_OPERATOR:
                if num_hands == 1:
                    detected_value = get_operator_gesture(results.multi_hand_landmarks[0])
                else:
                    error_message = "Show operator with only ONE hand"

        # --- State Machine Logic ---
        prompt = ""
        equation_str = ""

        if current_state == STATE_GET_FIRST_NUMBER:
            prompt = "Show the first number (0-10)"
            if detected_value is not None and detected_value == last_detected_value:
                gesture_counter += 1
                if gesture_counter >= GESTURE_CONFIRMATION_FRAMES:
                    first_number = detected_value
                    speak(f"First number is {first_number}")
                    current_state = STATE_GET_OPERATOR
                    gesture_counter = 0
                    last_detected_value = None
            else:
                gesture_counter = 0
                last_detected_value = detected_value
        
        elif current_state == STATE_GET_OPERATOR:
            prompt = "Show operator: (+, -, x, /)"
            equation_str = f"{first_number}"
            if detected_value is not None and detected_value == last_detected_value:
                gesture_counter += 1
                if gesture_counter >= GESTURE_CONFIRMATION_FRAMES:
                    operator = detected_value
                    op_name = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}.get(operator, "")
                    speak(op_name)
                    current_state = STATE_GET_SECOND_NUMBER
                    gesture_counter = 0
                    last_detected_value = None
            else:
                gesture_counter = 0
                last_detected_value = detected_value

        elif current_state == STATE_GET_SECOND_NUMBER:
            prompt = "Show the second number (0-10)"
            equation_str = f"{first_number} {operator}"
            if detected_value is not None and detected_value == last_detected_value:
                gesture_counter += 1
                if gesture_counter >= GESTURE_CONFIRMATION_FRAMES:
                    second_number = detected_value
                    speak(f"Second number is {second_number}")
                    
                    # --- Calculation ---
                    try:
                        if operator == '+':
                            result = first_number + second_number
                        elif operator == '-':
                            result = first_number - second_number
                        elif operator == '*':
                            result = first_number * second_number
                        elif operator == '/':
                            if second_number == 0:
                                result = "Error: Div by 0 Error"
                            else:
                                result = round(first_number / second_number, 2)
                        
                        result_speech = f"{equation_str} {second_number} equals {result}"
                        speak(result_speech)

                    except Exception as e:
                        result = "Error"
                        speak("An error occurred during calculation.")
                        print(f"Calculation Error: {e}")

                    current_state = STATE_SHOW_RESULT
                    result_display_start_time = time.time()
                    gesture_counter = 0
                    last_detected_value = None
            else:
                gesture_counter = 0
                last_detected_value = detected_value
        
        elif current_state == STATE_SHOW_RESULT:
            prompt = "Result. Resetting the calculator..."
            equation_str = f"{first_number} {operator} {second_number} = "
            
            # Reset after a short delay (1 second)
            if time.time() - result_display_start_time > 1:
                # Reset all variables
                first_number, operator, second_number, result = None, None, None, None
                current_state = STATE_GET_FIRST_NUMBER
                speak("Calculator reset.")

        # --- Display UI ---
        result_display = str(result) if result is not None else ""
        gesture_progress = gesture_counter / GESTURE_CONFIRMATION_FRAMES if gesture_counter else 0
        op_sym = operator if current_state == STATE_GET_OPERATOR else None
        frame = draw_ui(
            frame, 
            prompt, 
            equation_str,
            result_display if current_state == STATE_SHOW_RESULT else "",
            error_message,
            gesture_progress,
            op_sym
        )
        # Show the current detected value for feedback
        if last_detected_value is not None and current_state != STATE_SHOW_RESULT:
            feedback_text = f"Detecting: {last_detected_value}"
            cv2.putText(frame, feedback_text, (w - 300, 40), FONT, 1, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('Gesture Calculator', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()
    