# Krish_AI-ML-2_BLINKMATCH
🔐 BlinkMatch - Smart AI Face Recognition with Blink Detection
👥 Team Members
Chirag Sharma (2401730183)
Dev Yadav (2401730237)
Krish (2401730158)
Saurav (2401730213)
Group Supervisor: Dr. Ravinder Beniwal 📝 Short Project Description
BlinkMatch is an AI-powered facial recognition system that enhances security through liveness detection using blink verification.
Built using Python, OpenCV, and Streamlit, the system ensures that only real, live individuals can unlock access, effectively preventing spoofing via images or videos.
The system can be used in secure access control, healthcare identity verification, and retail automation.

--- 📺 Link to Video Explanation 📹 [https://github.com/krishkasnia00/Krish_AI-ML-2_BLINKMATCH/blob/main/BlinkMatch%20Project%20Video%20Presentation.mp4]

⚙️ Technologies Used

Technology	Purpose
Python	Core programming language
Streamlit	Web-based UI interface
OpenCV (cv2)	Image processing and webcam integration
face_recognition	Face detection, encoding, and comparison
Numpy	Array manipulation and math operations
Pickle	Save/load face encodings
Pandas	Manage and export entry logs
Datetime & Time	Timestamping and performance tracking
scipy.spatial.distance	Eye Aspect Ratio (EAR) calculation for blink detection
OS	File management operations
▶️ Steps to Run / Execute the Project

📦 1. Install Requirements Ensure you have Python installed. Then, install the required libraries:

pip install streamlit opencv-python face_recognition numpy pandas scipy

2. Run the application
streamlit run main.py

3. Register a Face
Go to the sidebar in the Streamlit app

Enter a user name

Capture an image using the webcam

Confirm and save the face encoding

4. Start Face Recognition + Blink Detection
Click the "Toggle Recognition" button

Blink in front of the camera

If recognized and blink is detected, the system will unlock and log the entry

5. View and Export Entry Logs
Go to the Entry Records tab

Filter records by name or date

Click Export to download logs as a CSV file

6. Optional: Clear Data
Use the sidebar to delete all registered face data

You can also clear the entry log entirely if needed
