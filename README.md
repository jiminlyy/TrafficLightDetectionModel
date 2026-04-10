Traffic Light Detection

Project Overview

This project is a traffic light detection model developed to recognize traffic light signals from images or video streams.
The main goal is to detect traffic lights accurately and provide a foundation for applications such as pedestrian assistance, safe walking support, and real-time traffic signal recognition.

This model was built using a YOLO-based object detection approach and trained on a custom traffic light dataset.
Through this project, I explored the full pipeline of an object detection task, including data collection, labeling, model training, validation, and inference.

⸻

Motivation

Traffic lights are one of the most important visual cues for pedestrians especially for blind people. 
This project is intended to support visually impaired pedestrians.
In particular, recognizing pedestrian signals correctly can be useful for building assistive systems for safer road crossing.

I started this project to understand how computer vision can be applied to real-world safety problems, and to gain hands-on experience with training a custom object detection model.

⸻

Features
	•	Detect traffic lights in video by smartphone camera
	•	Train a custom object detection model with labeled data
	•	Apply YOLO-based real-time detection
	•	Build a foundation for future pedestrian safety applications

⸻

Tech Stack
	•	Python
	•	YOLO / Ultralytics
	•	OpenCV
	•	Roboflow or custom labeled dataset
	•	Google Colab / local environment
	•	GitHub for version control

⸻

Dataset

The model was trained on a custom dataset prepared for traffic light detection.
The dataset includes labeled images of traffic lights collected and organized for object detection training.

Data Preparation Process
	1.	Collected traffic light image data from RoboFlow open dataset
	2.	Labeled objects with bounding boxes
	3.	Organized the dataset into training/validation sets
	4.	Exported the dataset in YOLO format
	5.	Trained the model using the dataset configuration file

⸻

Model Training

The model was trained using a YOLO-based architecture.

Example training process:
	•	Base model: yolov8n.pt
	•	Input size: 640
	•	Epochs: 50
	•	Dataset config: data.yaml

This project helped me understand:
	•	how object detection models learn from labeled bounding box data,
	•	how training performance changes over epochs,
	•	and how dataset quality affects detection accuracy.

⸻

Inference

After training, the model can be used to detect traffic lights from:
	•	single images
	•	recorded videos
	•	real-time camera input

The prediction result includes bounding boxes around detected traffic lights and confidence scores for each detection.

⸻

Project Structure
trafficLightDetection/
├── dataset/
│   ├── train/
│   ├── valid/
│   └── data.yaml
├── runs/
│   └── detect/
├── models/
├── scripts/
├── best.pt
└── README.md


⸻

What I Learned

Through this project, I learned:
	•	how to prepare a custom dataset for object detection,
	•	how to train and evaluate a YOLO model,
	•	how to manage a computer vision project from data collection to deployment preparation,
	•	and how object detection can be applied to real-world assistive technology.

⸻

Limitations

This project still has several limitations:
	•	detection performance may vary depending on lighting conditions and image quality,
	•	small or distant traffic lights can be harder to detect,
	•	the current model focuses on detection and does not fully interpret complex traffic signal logic.

These limitations suggest directions for future improvement.
=> That's why I couldn't use for my project. We were supposed that a person hanging their phone in front of them.
And the app anaylize Traffic Light. But the Traffic Light will be small so the app can't caught it.

⸻

Future Work
	•	Improve accuracy with more diverse training data
	•	Distinguish between pedestrian signals and vehicle signals more clearly
	•	Optimize the model for mobile deployment
	•	Connect the detection model to a real-time guidance system
	•	Combine detection with signal state recognition and remaining time prediction

⸻

Conclusion

This project is a practical attempt to apply deep learning-based object detection to traffic light recognition.
It is meaningful not only as a computer vision practice project, but also as a step toward building safety-focused AI applications for everyday life.

⸻
