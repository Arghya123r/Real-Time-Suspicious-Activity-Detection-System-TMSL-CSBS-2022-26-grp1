from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime, timedelta

app = Flask(__name__)

CSV_FILE = 'timestamps.csv'
MERGED_CSV_FILE = 'merged.csv'
HEADERS = ['video_id', 'start_time', 'end_time']
MERGED_HEADERS = ['video_id', 'start_time', 'end_time']

def time_to_seconds(time_str):
    """Convert HH:MM:SS.ms to seconds"""
    try:
        # Handle HH:MM:SS.ms format
        if '.' in time_str:
            time_part = time_str.split('.')[0]  # Remove milliseconds
        else:
            time_part = time_str
        h, m, s = time_part.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        raise ValueError(f"Invalid time format: {time_str}")

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS.ms"""
    td = datetime.utcfromtimestamp(seconds)
    return td.strftime('%H:%M:%S.%f')[:-3]

def merge_timestamps_for_video(intervals):
    """Merge intervals if gap < 5 minutes (300 seconds)"""
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: time_to_seconds(x['start_time']))
    merged = [intervals[0].copy()]  # Copy first interval
    
    for current in intervals[1:]:
        last = merged[-1]
        last_end_sec = time_to_seconds(last['end_time'])
        curr_start_sec = time_to_seconds(current['start_time'])
        
        if curr_start_sec - last_end_sec <= 300:  # checking if the time gap is less than or equal to 5 minutes
            # Extend end time if current ends later
            if time_to_seconds(current['end_time']) > last_end_sec:
                last['end_time'] = current['end_time']
        else:
            merged.append(current.copy())
    
    return merged

@app.route('/merge_timestamps', methods=['POST'])
def merge_timestamps():
    if not os.path.exists(CSV_FILE):
        return jsonify({'error': 'No timestamps.csv file found'}), 404
    
    # Read all timestamps
    timestamps = []
    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamps.append(row)
    
    if not timestamps:
        return jsonify({'message': 'No timestamps to merge'}), 200
    
    # Group by video_id
    video_groups = {}
    for row in timestamps:
        video_id = row['video_id']
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(row)
    
    # Merge timestamps for each video
    merged_data = []
    for video_id, intervals in video_groups.items():
        merged_intervals = merge_timestamps_for_video(intervals)
        for interval in merged_intervals:
            interval['video_id'] = video_id  # Ensure video_id is set
            merged_data.append(interval)
    
    # Save merged timestamps to new CSV
    with open(MERGED_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=MERGED_HEADERS)
        writer.writeheader()
        writer.writerows(merged_data)
    

if __name__ == '__main__':
    import sys
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app.run(debug=True)