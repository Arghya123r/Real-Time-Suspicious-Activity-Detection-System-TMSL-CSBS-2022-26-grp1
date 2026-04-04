# Real-Time Suspicious Activity Detection System
### TMSL-CSBS-2022-26 | Group-1

---

##  Project Overview
This research-oriented project implements an automated surveillance system designed to identify anomalous human behaviors (e.g., violence, robbery) in video feeds. Utilizing a **Deep Learning (LRCN)** architecture, the system transforms raw video data into actionable security insights by logging detection timestamps and triggering real-time alerts.

---

##  System Architecture & Modules

| Module | Component | Primary Function |
| :--- | :--- | :--- |
| **ML Engine** | `ActivityDetector.py` | Acts as the primary inference engine utilizing a **Long-term Recurrent Convolutional Network (LRCN)** to classify actions. |
| **Data Persistence** | `Api_to store.py` | Captures raw detection data via POST requests and logs them into a structured `timestamps.csv` in `HH:MM:SS.ms` format. |
| **Event Merging** | `merge_api_2.py` | Implements interval-merging logic to group fragmented detections into single events if the gap is **≤ 300 seconds**. |
| **Alert System** | `Alert system.py` | Fetches contacts from **MongoDB Atlas** and dispatches **SMTP** email alerts with localized **IST timestamps**. |

---

##  Getting Started

### Prerequisites
* **Dataset**: Download the [UCF-Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset).
* **Environment Setup**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

###  Training Pipeline
Follow these steps to train the detection model on your local machine:

1.  **Open Notebook**: Navigate to and open the training environment:
    `Model_Code/training-book-lightweight.ipynb`
2.  **Update Configuration**: 
    * Locate the configuration section within the notebook.
    * Update the `dataset_root` or dataset path to point to your local **UCF-Crime** directory.
3.  **Execute Training**: 
    * Run all cells in the notebook to initiate the training process.
    * Once complete, the system will generate the trained weights file (e.g., `lrcn_ucf_best.pt`).

###  Inference & Execution
1.  **Deploy**: Place the trained model file (`lrcn_ucf_best.pt`) into the `Model/` directory.
2.  **Run**: Execute the detector via the terminal:
    ```bash
    python ActivityDetector.py --input <path_to_video_file>
    ```
3.  **Results**: Processed outputs and logs are stored in the `output/` folder.

---

##  System Information & Limitations
* **Processing Mode**: Currently optimized for file-based processing; live stream and webcam support are disabled for system stability.
* **Database Management**: User contact information for alerts is securely managed via **MongoDB Atlas**.
* **Clip Generation**: Automated video summaries are generated based on the optimized `merged.csv` output.

---



