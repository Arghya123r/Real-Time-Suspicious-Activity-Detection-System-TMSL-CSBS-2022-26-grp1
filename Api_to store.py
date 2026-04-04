from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime


app = Flask(__name__)


CSV_FILE = 'timestamps.csv'
HEADERS = ['video_id', 'start_time', 'end_time']


@app.route('/store_timestamp', methods=['POST'])
def store_timestamp():
    data = request.get_json()
    
    # Validate parameters
    if not all(key in data for key in ['video_id', 'start_time', 'end_time']):
        return jsonify({'error': 'Missing required parameters: video_id, start_time, end_time'}), 400
    
    video_id = data['video_id']
    
    try:
        # Parse input times (expects seconds as float, e.g., "10.5")
        start_time = float(data['start_time'])
        end_time = float(data['end_time'])
        
        # Convert seconds to HH:MM:SS.ms format
        start_time_str = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S.%f')[:-3])
        end_time_str = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3])
        
    except (ValueError, TypeError):
        return jsonify({'error': 'start_time and end_time must be valid numbers (seconds)'}), 400
    
    file_exists = os.path.exists(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'video_id': video_id,
            'start_time': start_time_str,
            'end_time': end_time_str
        })
    


if __name__ == '__main__':
    app.run(debug=True)