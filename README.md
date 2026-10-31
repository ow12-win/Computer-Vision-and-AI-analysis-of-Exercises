# Gym AI Tracker

### AI-Powered Exercise Tracking and Performance Analysis

## Overview

Gym AI Tracker is a Python-based fitness tracking application that I developed to explore how artificial intelligence, computer vision and machine learning can be used to analyse exercise performance in real time.

The application uses a webcam to detect body movements, identify exercises and count repetitions automatically. It also calculates performance metrics such as joint angles, movement velocity, estimated force, power, fatigue and repetition quality.

The aim of this project was to create a system that could make gym training more data-driven without requiring expensive wearable devices or specialist equipment.

This project allowed me to combine my interests in fitness, programming and artificial intelligence while developing my understanding of how machine learning systems work in real-world applications.

## Technologies Used

- **Python** – Main programming language.
- **OpenCV** – Capturing and processing live webcam footage.
- **MediaPipe** – Detecting body landmarks and tracking human poses.
- **NumPy** – Mathematical calculations involving joint angles.
- **Scikit-learn** – Training a machine learning model to classify exercises.
- **Pandas** – Loading and processing training datasets.
- **PyQt6** – Building a graphical user interface.
- **Joblib** – Saving and loading the trained machine learning model.
- **JSON and CSV** – Storing workout information and exercise training data.

## Development Process

### Stage 1 – Research and Planning

I started by researching how computer vision could be used to track human movement.

I wanted to understand how a computer could recognise different exercises using only a webcam.

I explored pose estimation and discovered MediaPipe, which can detect key body landmarks such as shoulders, elbows, hips, knees and ankles.

I planned the project around three main objectives:

1. Detect body movements using a webcam.
2. Count exercise repetitions automatically.
3. Analyse exercise performance using movement data.

### Stage 2 – Building the Camera System

The first part of the development process was creating a basic webcam application using OpenCV.

I used Python to access the camera, capture individual frames and display the live video feed.

I then experimented with PyQt6 to create a separate graphical interface where the camera feed could be displayed inside an application window.

This helped me understand how to work with live video, image formats and graphical user interfaces.

### Stage 3 – Implementing Pose Detection

Once the camera system was working, I integrated MediaPipe Pose.

MediaPipe allowed me to identify body landmarks in each video frame.

I used these landmarks to track different parts of the body during exercise movements.

For example, during a squat, the system uses the hip, knee and ankle landmarks to calculate the knee angle.

I created a separate Python file called `angle_helper.py` to calculate angles between three points.

This made the code easier to organise and allowed the same function to be reused across different exercises.

### Stage 4 – Creating the Repetition Counter

After implementing pose detection, I developed a repetition-counting system.

The program uses joint angles to determine when a person reaches different stages of an exercise.

For example, during a squat, the system detects when the knees bend and when the person returns to a standing position.

I introduced angle thresholds to recognise completed repetitions.

One challenge was preventing the system from counting the same repetition multiple times.

To improve this, I implemented movement stages, smoothing techniques and a cooldown period between repetitions.

I also added separate tracking for the left and right sides of the body.

### Stage 5 – Collecting Exercise Data

The next stage was preparing data for machine learning.

I created a system that records movement features and stores them in a CSV dataset.

The features include:

- Knee angle
- Hip angle
- Elbow angle
- Shoulder angle
- Wrist velocity
- Hip velocity

Each data sample is assigned a label representing the exercise or movement being performed.

This dataset provides the information needed to train a machine learning model to recognise different exercises.

### Stage 6 – Training the Machine Learning Model

I used Scikit-learn to build an exercise classification model.

I selected a Random Forest Classifier because it can identify patterns in numerical data and classify different types of movement.

The dataset was divided into training and testing sets using an 80/20 split.

The model was trained using the extracted movement features and their corresponding exercise labels.

I also implemented model evaluation using accuracy, a classification report and a confusion matrix.

Once trained, the model was saved as `exercise_model.pkl` using Joblib so it could be loaded into the main application.

### Stage 7 – Adding Automatic Exercise Detection

After developing the classifier, I integrated it into the live exercise tracking system.

The application extracts movement features from the webcam and sends them to the trained model.

The model predicts which exercise is being performed.

I added a confidence threshold and prediction smoothing to reduce unstable classifications.

In AI Auto-Detect mode, the program can automatically change its tracking configuration based on the predicted exercise.

The system includes configurations for exercises such as:

- Squats
- Conventional deadlifts
- Sumo deadlifts
- Bench presses
- Bicep curls
- Shoulder presses
- Lateral raises
- Lunges

I also included a custom exercise option where users can define their own joint landmarks and angle thresholds.

### Stage 8 – Implementing Performance Analytics

Once the exercise detection and repetition tracking were developed, I wanted to make the project more useful by adding performance analysis.

I created a system that calculates metrics including:

- Range of motion
- Repetition duration
- Movement tempo
- Angular velocity
- Estimated force
- Estimated power
- Velocity loss
- Fatigue index
- Estimated Rate of Perceived Exertion (RPE)
- Repetition quality score

I used movement measurements and simplified biomechanics calculations to estimate force and power.

I also implemented a Kalman filter to smooth noisy joint-angle measurements and estimate angular velocity.

These additions helped me explore how mathematics and physics can be combined with computer vision to analyse exercise performance.

The force and power calculations are approximations based on movement assumptions rather than direct measurements from specialist equipment.

### Stage 9 – Building the Interface and Workout Tracking

I developed graphical interface components to make the application easier to interact with.

The project includes a PyQt6 interface prototype with controls for moving to the next set, resetting the workout and closing the application.

The main tracking application also displays information directly over the webcam footage, including repetition counts, exercise predictions, movement angles and performance metrics.

I added workout set tracking and a rest timer.

At the end of a session, the program saves workout information into `exercise_data.json`.

This allows the collected information to be reviewed after training.

### Stage 10 – Testing and Improvements

Throughout development, I worked on improving the reliability of the tracking system.

Some of the main challenges involved noisy landmark measurements, inaccurate repetition counting and changes in camera positioning.

I introduced several techniques to help address these problems, including:

- Angle smoothing
- Kalman filtering
- Confidence thresholds
- Prediction smoothing
- Repetition cooldowns
- Exercise-specific movement thresholds
- Checks for missing or low-visibility landmarks

I also experimented with different approaches to counting repetitions for different exercises.

For example, deadlift tracking uses both hip extension and upward wrist movement to help recognise the completion of a repetition.

## Project File Structure

```text
Gym-AI-Tracker/
│
├── main.py
├── angle_helper.py
├── exercise_classifier.py
├── GUI.py
├── camera_gui.py
│
├── dataset.csv
├── exercise_model.pkl
├── exercise_data.json
│
└── README.md
```

### File Descriptions

**main.py** – Main exercise tracking application, including pose detection, automatic classification, repetition counting and performance analytics.

**angle_helper.py** – Reusable function for calculating joint angles from body landmarks.

**exercise_classifier.py** – Trains and evaluates the Random Forest exercise classification model.

**GUI.py** – PyQt6 exercise tracking interface with repetition and set controls.

**camera_gui.py** – Basic webcam interface prototype.

**dataset.csv** – Dataset containing movement features and exercise labels.

**exercise_model.pkl** – Saved machine learning classification model.

**exercise_data.json** – Example workout session data containing repetitions, sets and performance metrics.

## Installation and Running the Project

### Requirements

- Python
- A working webcam
- Required Python libraries

Install the dependencies:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib PyQt6
```

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Gym-AI-Tracker.git
```

Navigate into the project folder:

```bash
cd Gym-AI-Tracker
```

To train or retrain the classification model:

```bash
python exercise_classifier.py
```

To run the main tracking application:

```bash
python main.py
```

The application will ask for your bodyweight, external load and preferred exercise tracking mode before opening the webcam.

The supplied model can also be used without retraining, provided its dependencies are compatible.

## What I Learned

### 1. Computer Vision

I learned how computer vision systems process webcam footage and identify body landmarks in real time.

I gained experience using OpenCV and MediaPipe to turn visual information into numerical data that can be analysed programmatically.

### 2. Machine Learning

I developed my understanding of supervised machine learning by working with labelled exercise data.

I learned how to prepare features, split datasets, train a Random Forest model and evaluate predictions.

I also learned how to save a trained model and integrate it into another Python application.

### 3. Mathematics and Biomechanics

I improved my understanding of how mathematics can be applied to human movement.

I used joint-angle calculations, velocity measurements and simplified physics equations to estimate exercise performance.

This helped me understand the relationship between movement data and physical performance.

### 4. Real-Time Programming

Working with a live webcam taught me how important efficient processing is when an application needs to respond continuously.

I learned how to capture video frames, process pose landmarks and update information while the program is running.

### 5. Data Collection and Processing

I learned how to collect numerical data, organise it into a CSV dataset and use it for machine learning.

I also gained experience saving structured workout information in JSON format.

### 6. Graphical User Interface Development

I gained experience using PyQt6 to build desktop application interfaces.

I learned how to display live camera footage, create interactive buttons and connect user actions to Python functions.

### 7. Problem-Solving and Debugging

One of the most valuable parts of the project was learning how to break a complex problem into smaller stages.

Instead of trying to build the entire system at once, I worked on individual components such as camera capture, pose estimation, repetition counting and classification.

This approach helped me develop my debugging skills and understand how different parts of a software project interact.

## Challenges Faced

One of the biggest challenges was making repetition counting reliable.

Body landmark positions can fluctuate between frames, causing joint angles to change even when the user is relatively still.

To reduce this problem, I introduced smoothing and filtering techniques.

Another challenge was recognising different exercises using movement data. Different exercises can produce similar joint angles, which makes classification more difficult.

This helped me understand the importance of collecting varied, accurately labelled training data.

I also found that combining multiple technologies into one application required careful organisation, particularly when integrating computer vision, machine learning and graphical interfaces.

## Future Improvements

There are several features I would like to explore in future versions:

- Improve exercise classification using a larger and more varied dataset.
- Develop a more polished and fully integrated graphical interface.
- Add exercise form feedback and posture correction.
- Create workout history graphs to show progress over time.
- Improve force and power estimates using calibrated movement measurements.
- Add support for more exercises.
- Explore deep learning models for movement recognition.
- Develop a mobile version of the application.
- Test the system with different camera angles, lighting conditions and users.

## Conclusion

Gym AI Tracker has been a valuable project for developing my programming and problem-solving skills.

It allowed me to apply Python, computer vision, machine learning and mathematics to a practical problem.

The project also helped me understand how complex applications can be developed by building and testing smaller components before combining them into a larger system.

Most importantly, it gave me practical experience working with real-time data and integrating artificial intelligence into a working software application.

This project has encouraged me to continue exploring artificial intelligence, software engineering and the development of intelligent applications.
