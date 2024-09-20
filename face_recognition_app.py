import cv2
import face_recognition
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = 'secret_key'

# Initialize variables to store registered face encodings and their names
known_face_encodings = []
known_face_names = []

# Directory to store registered face images
REGISTRATION_DIR = 'registered_faces'
if not os.path.exists(REGISTRATION_DIR):
    os.makedirs(REGISTRATION_DIR)

# Function to generate frames for the recognition feed
def gen_recognition_feed():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Convert BGR to RGB for face_recognition
            rgb_frame = frame[:, :, ::-1]
            
            # Find all faces and their encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = None
                if matches:
                    best_match_index = matches.index(True)
                
                if best_match_index is not None and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                # Draw a box around the face and label with name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    video_capture.release()

# Function to register a face
@app.route('/register', methods=['POST'])
def register():
    if 'file' not in request.files or 'name' not in request.form:
        flash('No file or name provided!')
        return redirect(request.url)

    file = request.files['file']
    name = request.form['name']
    
    if file.filename == '':
        flash('No file selected!')
        return redirect(request.url)
    
    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        # Save the uploaded image
        image_path = os.path.join(REGISTRATION_DIR, f"{name}.jpg")
        file.save(image_path)
        
        # Load the captured image and encode the face
        try:
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            
            if len(face_encoding) == 0:
                flash('No face detected in the image!')
                return redirect(request.url)
            
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(name)
            
            flash(f'Registered {name} successfully!')
        except Exception as e:
            flash(f'Error: {str(e)}')
    else:
        flash('Invalid file format. Please upload a PNG, JPG, or JPEG image.')
    
    # After registration, keep the recognition feed active
    return redirect(url_for('index'))

# Route to display the recognition live feed
@app.route('/recognize_feed')
def recognize_feed():
    return Response(gen_recognition_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main route for the home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
