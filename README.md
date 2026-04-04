Real-Time Suspicious Activity Detection System (TMSL-CSBS-2022-26-Grp1)
📌 Overview
This project implements a Real-Time Suspicious Activity Detection System designed for automated surveillance. Using a Deep Learning (LRCN) architecture, the system identifies anomalous behaviors (e.g., violence, robbery) in video feeds, logs detection timestamps, and triggers real-time alerts for security personnel.

🛠️ System Architecture & Modules
1. ML Detection Engine (ActivityDetector.py)
The core inference engine that processes video input using a pre-trained LRCN (Long-term Recurrent Convolutional Network) model to classify actions in real-time.

2. Data Persistence Layer (Api_to store.py)
Function: Captures raw frame-level detection data.

Workflow: Receives video_id, start_time, and end_time via POST requests and logs them into a structured timestamps.csv.

Formatting: Automatically converts raw seconds into a precise HH:MM:SS.ms format for readable logs.

3. Temporal Event Merging (merge_api_2.py)
Function: Optimizes detection logs for human review.

Algorithm: Implements an interval-merging logic where fragmented detections are combined into a single event if the time gap is less than 5 minutes (300 seconds).

Output: Generates a refined merged.csv which serves as the source for video clip generation.

4. Intelligent Alert System (Alert system.py)
Function: Automated notification dispatch.

Connectivity: Integrated with MongoDB to fetch registered security personnel emails.

Mechanism: Uses SMTP to send instant email alerts with localized (IST) timestamps when suspicious activity is detected.

🚀 Getting Started
Prerequisites
Dataset: Download the UCF-Crime Dataset.

Environment: ```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Training
Navigate to Model_Code/training-book-lightweight.ipynb.

Update the dataset_root variable in the cfg class with your local Train directory path.

Execute all cells to train the model.

Inference
Place the trained model (lrcn_ucf_best.pt) inside the Model/ folder.

Run the detector via terminal:

Bash
python ActivityDetector.py --input <path_to_video_file>
View results in the output/ folder.

📋 Information & Limitations
Real-time Processing: Live stream and webcam views are currently disabled for stability; the system currently supports file-based processing.

Database: Contact info is managed via MongoDB Atlas.

Video Generation: Refined clips are generated based on the merged.csv output.

👥 Contributors
Arghya: ML Model Development & Dataset Training.

Soumadityo: Alert Systems, API Development, and MongoDB Integration.

Utsab: Backend Authentication, SQLAlchemy Database Schema, and System Integration.

Shreyashi: Video Player UI and Frontend Logic.




