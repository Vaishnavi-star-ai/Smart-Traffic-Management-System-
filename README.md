# 📖 Smart Traffic Management System 🚦

🚀 Overview
Smart Traffic Management System is an AI-powered solution designed to optimize traffic flow and enhance safety at intersections. The system integrates advanced computer vision and deep learning techniques to:

✅ Prioritize emergency vehicles like ambulances.
✅ Detect and recognize number plates for enforcement or analysis.
✅ Monitor specific areas of interest to manage congestion or incidents.
Built with YOLOv8 fine-tuned for high accuracy and real-time inference, this project is seamlessly integrated with a MySQL database for data logging, querying, and long-term analytics.

🎯 Features
🏥 Ambulance Detection & Priority: Automatic recognition and prioritization of emergency vehicles.

🔍 Number Plate Detection: Capture and log number plates for analytics and enforcement.

🎥 Areas of Interest Monitoring: Specify and track custom regions for congestion, violations, or incident detection.

🧠 YOLOv8 Model: Fine-tuned and optimized for increased detection accuracy.

🗄️ MySQL Integration: Scalable and structured database to store detected data.

⏱️ Real-Time Processing: Efficient and low-latency processing suitable for real-world deployment.

🧰 Tech Stack
Component      	Technology
Computer         Vision	YOLOv8 (Ultralytics)
Backend	        Python (FastAPI/Flask)
Database	      MySQL
Deployment	    Docker, Linux
board           raspberry pi 5

📝 Installation
Follow these steps to set up the project:
# Clone this repository
git clone http://github.com/Rohith0750/Smart-Traffic-System/tree/main
cd smart-traffic-management

# Install dependencies
pip install -r requirements.txt

# Set up the MySQL database
# Import the schema
mysql -u your_user -p traffic_mangaement < database/schema.sql

# Configure your DB credentials in config.py


🧠 Model Training
This project uses a YOLOv8 model fine-tuned on a custom dataset of ambulances and license plates.

# Train the model
yolo train data=dataset.yaml model=yolov8s.pt epochs=100

# Evaluate the model
yolo val model=runs/train/weights/best.pt   data=dataset.yaml



📊 Performance
Metric	              Value
mAP (50-95)         	0.89
Inference Speed	     25 FPS on GPU
Precision	           0.92
Recall	             0.90

💡 Contributing
Contributions, bug reports, and feature suggestions are always welcome.
Feel free to submit a pull request or raise an issue!


# Happy Coding! 🎉
