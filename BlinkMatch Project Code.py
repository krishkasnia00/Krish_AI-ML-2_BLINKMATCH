import streamlit as st
import cv2
import face_recognition
import numpy as np
import time
import pickle
import os
import pandas as pd
from scipy.spatial import distance as dist
from datetime import datetime

st.set_page_config(
    page_title="Smart AI Face Door Lock",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Eye aspect ratio function to detect blinks
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    # Return the eye aspect ratio
    return ear

FACE_DATA_FILE = "face_recognition_data.pkl"
ENTRY_LOG_FILE = "entry_log.csv"

def load_face_data():
    if os.path.exists(FACE_DATA_FILE):
        try:
            with open(FACE_DATA_FILE, 'rb') as file:
                data = pickle.load(file)
                return data['encodings'], data['names']
        except Exception as e:
            st.error(f"Error loading saved face data: {e}")
    return [], []

def save_face_data(encodings, names):
    try:
        with open(FACE_DATA_FILE, 'wb') as file:
            pickle.dump({'encodings': encodings, 'names': names}, file)
        return True
    except Exception as e:
        st.error(f"Error saving face data: {e}")
        return False

def load_entry_log():
    if os.path.exists(ENTRY_LOG_FILE):
        try:
            return pd.read_csv(ENTRY_LOG_FILE)
        except Exception as e:
            st.error(f"Error loading entry log: {e}")
    return pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])

def save_entry_log(entry_log):
    try:
        entry_log.to_csv(ENTRY_LOG_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving entry log: {e}")
        return False

def log_entry(name, status="Entered"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Load existing log
    entry_log = load_entry_log()
    
    # Add new entry
    new_entry = pd.DataFrame({
        'Name': [name],
        'Date': [date_str],
        'Time': [time_str],
        'Status': [status]
    })
    
    entry_log = pd.concat([new_entry, entry_log], ignore_index=True)
    
    # Save updated log
    save_entry_log(entry_log)
    
    return entry_log

if 'app_initialized' not in st.session_state:
    known_encodings, known_names = load_face_data()
    entry_log = load_entry_log()

    st.session_state.app_initialized = True
    st.session_state.known_face_encodings = known_encodings
    st.session_state.known_face_names = known_names
    st.session_state.entry_log = entry_log
    st.session_state.lock_status = "locked"
    st.session_state.recognized_names = []
    st.session_state.registration_step = 0
    st.session_state.current_name = ""
    st.session_state.temp_image = None
    st.session_state.recognition_active = False
    
    # Add blink detection related variables
    st.session_state.blink_detected = False
    st.session_state.blink_counter = 0
    st.session_state.ear_threshold = 0.2  # EAR threshold for blink detection
    st.session_state.consec_frames = 3    # Number of consecutive frames to consider a blink
    
    # Entry logging variables
    st.session_state.last_log_time = {}  # Dictionary to track last entry time for each person

st.title("AI Face Recognition Door Lock System")

tab1, tab2 = st.tabs(["Access Control", "Entry Records"])

with tab1:
    st.header("Face Recognition")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("<h3 style='text-align: center;'>Lock Status</h3>", unsafe_allow_html=True)
        lock_status_container = st.empty()
        welcome_message = st.empty()
        blink_status = st.empty()

        if st.session_state.lock_status == "locked":
            lock_status_container.markdown("<h1 style='text-align: center; color: red;'>🔒</h1>", unsafe_allow_html=True)
        else:
            lock_status_container.markdown("<h1 style='text-align: center; color: green;'>🔓</h1>", unsafe_allow_html=True)
            if st.session_state.recognized_names:
                welcome_message.markdown(f"<p style='text-align: center;'>Welcome, {', '.join(set(st.session_state.recognized_names))}!</p>", unsafe_allow_html=True)

    frame_placeholder = col1.empty()

    if st.button("Toggle Recognition", key="toggle_button"):
        st.session_state.recognition_active = not st.session_state.recognition_active

    st.write(f"Recognition: {'Active' if st.session_state.recognition_active else 'Inactive'}")

with tab2:
    st.header("Entry Records")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_date = st.date_input("Filter by date", datetime.now())
    with col2:
        filter_name = st.selectbox("Filter by name", ["All"] + list(set(st.session_state.known_face_names)))
    
    # Load the latest entry log
    entry_log = load_entry_log()
    
    # Apply filters
    filtered_log = entry_log.copy()
    if filter_name != "All":
        filtered_log = filtered_log[filtered_log['Name'] == filter_name]
    
    filter_date_str = filter_date.strftime("%Y-%m-%d")
    filtered_log = filtered_log[filtered_log['Date'] == filter_date_str]
    
    # Display the filtered log
    st.dataframe(filtered_log, use_container_width=True)
    
    # Add export button
    if not filtered_log.empty:
        csv = filtered_log.to_csv(index=False)
        st.download_button(
            label="Export Records",
            data=csv,
            file_name=f"entry_log_{filter_date_str}.csv",
            mime="text/csv"
        )
    
    # Clear log button
    if st.button("Clear All Records"):
        if os.path.exists(ENTRY_LOG_FILE):
            os.remove(ENTRY_LOG_FILE)
        st.session_state.entry_log = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
        st.success("All entry records cleared successfully!")
        st.rerun()

st.sidebar.header("Register New Face")

if st.session_state.registration_step == 0:
    st.session_state.current_name = st.sidebar.text_input("Enter your name:")
    
    if st.sidebar.button("Next") and st.session_state.current_name:
        st.session_state.registration_step = 1
        st.rerun()

elif st.session_state.registration_step == 1:
    st.sidebar.info(f"Registering: {st.session_state.current_name}")
    img_file_buffer = st.sidebar.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_rgb)

        if len(face_locations) > 0:
            st.session_state.temp_image = img_rgb
            st.session_state.registration_step = 2
            st.rerun()
        else:
            st.sidebar.error("No face detected. Please try again.")
            st.session_state.registration_step = 0
            st.rerun()

elif st.session_state.registration_step == 2:
    st.sidebar.info(f"Registering: {st.session_state.current_name}")
    st.sidebar.image(st.session_state.temp_image, caption="Captured Image", use_column_width=True)

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Confirm"):
        face_locations = face_recognition.face_locations(st.session_state.temp_image)
        face_encoding = face_recognition.face_encodings(st.session_state.temp_image, face_locations)[0]

        st.session_state.known_face_encodings.append(face_encoding)
        st.session_state.known_face_names.append(st.session_state.current_name)

        save_success = save_face_data(st.session_state.known_face_encodings, st.session_state.known_face_names)

        st.session_state.registration_step = 0
        st.session_state.current_name = ""
        st.session_state.temp_image = None

        if save_success:
            st.sidebar.success("Registration successful and data saved!")
        else:
            st.sidebar.success("Registration successful, but data could not be saved to disk.")

        st.rerun()

    if col2.button("Retake"):
        st.session_state.registration_step = 1
        st.rerun()

if st.session_state.known_face_names and st.sidebar.button("Delete All Face Data"):
    if os.path.exists(FACE_DATA_FILE):
        os.remove(FACE_DATA_FILE)
    st.session_state.known_face_encodings = []
    st.session_state.known_face_names = []
    st.sidebar.success("All face data deleted successfully!")
    st.rerun()

st.sidebar.header("Registered Faces")
if st.session_state.known_face_names:
    for name in st.session_state.known_face_names:
        st.sidebar.text(f"✓ {name}")
else:
    st.sidebar.text("No faces registered yet.")

# Add blink detection settings
st.sidebar.header("Blink Detection Settings")
st.session_state.ear_threshold = st.sidebar.slider("Eye Aspect Ratio Threshold", 0.1, 0.4, 0.2, 0.01)
st.session_state.consec_frames = st.sidebar.slider("Consecutive Frames for Blink", 1, 5, 3, 1)

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    current_recognized_names = []
    blink_detected = False
    
    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks):
        # Check for face match
        matches = face_recognition.compare_faces(st.session_state.known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        if True in matches and len(st.session_state.known_face_names) > 0:
            match_index = matches.index(True)
            name = st.session_state.known_face_names[match_index]
            current_recognized_names.append(name)
        
        # Draw face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Eye detection and blink detection
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Average EAR of both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Draw eye contours and bounding boxes
            left_eye_hull = cv2.convexHull(np.array(left_eye))
            right_eye_hull = cv2.convexHull(np.array(right_eye))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 255), 1)
            
            # Get eye bounding boxes
            left_eye_points = np.array(left_eye)
            right_eye_points = np.array(right_eye)
            
            left_x_min, left_y_min = np.min(left_eye_points, axis=0)
            left_x_max, left_y_max = np.max(left_eye_points, axis=0)
            right_x_min, right_y_min = np.min(right_eye_points, axis=0)
            right_x_max, right_y_max = np.max(right_eye_points, axis=0)
            
            # Draw eye bounding boxes
            cv2.rectangle(frame, (left_x_min, left_y_min), (left_x_max, left_y_max), (255, 0, 0), 2)
            cv2.rectangle(frame, (right_x_min, right_y_min), (right_x_max, right_y_max), (255, 0, 0), 2)
            
            # Check for blink
            if ear < st.session_state.ear_threshold:
                st.session_state.blink_counter += 1
            else:
                # If eyes were closed for enough frames, count as a blink
                if st.session_state.blink_counter >= st.session_state.consec_frames:
                    st.session_state.blink_detected = True
                    blink_detected = True
                # Reset counter
                st.session_state.blink_counter = 0
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (right, bottom + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Previously unrecognized names
    previously_recognized = set(st.session_state.recognized_names)
    
    # Update lock status based on face recognition AND blink detection
    if current_recognized_names and blink_detected:
        st.session_state.lock_status = "unlocked"
        
        # Log entries for newly recognized faces
        current_time = time.time()
        for name in current_recognized_names:
            # Only log if this person wasn't recognized before or if it's been more than 60 seconds
            if (name not in previously_recognized or 
                name not in st.session_state.last_log_time or 
                current_time - st.session_state.last_log_time.get(name, 0) > 60):
                
                # Update the entry log
                st.session_state.entry_log = log_entry(name)
                st.session_state.last_log_time[name] = current_time
    else:
        st.session_state.lock_status = "locked"
    
    st.session_state.recognized_names = current_recognized_names
    
    return frame, blink_detected

if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False

if st.session_state.recognition_active:
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam")
        else:
            st.success("Webcam is now active")
            
            while st.session_state.recognition_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                processed_frame, blink_detected = process_frame(frame)
                
                frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                
                if st.session_state.lock_status == "unlocked":
                    lock_status_container.markdown("<h1 style='text-align: center; color: green;'>🔓</h1>", unsafe_allow_html=True)
                    welcome_message.markdown(f"<p style='text-align: center;'>Welcome, {', '.join(set(st.session_state.recognized_names))}!</p>", unsafe_allow_html=True)
                    blink_status.markdown("<p style='text-align: center; color: green;'>Blink Detected ✓</p>", unsafe_allow_html=True)
                else:
                    lock_status_container.markdown("<h1 style='text-align: center; color: red;'>🔒</h1>", unsafe_allow_html=True)
                    welcome_message.empty()
                    if blink_detected:
                        blink_status.markdown("<p style='text-align: center; color: green;'>Blink Detected ✓</p>", unsafe_allow_html=True)
                    else:
                        blink_status.markdown("<p style='text-align: center; color: red;'>No Blink Detected</p>", unsafe_allow_html=True)
                
                time.sleep(0.1)
                
                if not st.session_state.recognition_active:
                    break
            
            cap.release()
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.recognition_active = False
else:
    frame_placeholder.info("Click 'Toggle Recognition' to start the webcam")







