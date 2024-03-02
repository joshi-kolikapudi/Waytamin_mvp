import cv2
import torch
import json
from datetime import datetime, timedelta
from multiprocessing import Process, Manager

class Tracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def add_track(self, centroid):
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = {
            "centroid": centroid,
            "first_seen": datetime.now(),
            "last_seen": datetime.now(),
            "consecutive_misses": 0  # Initialize consecutive misses counter
        }
        return track_id

    def update_track(self, track_id, centroid):
        self.tracks[track_id]["centroid"] = centroid
        self.tracks[track_id]["last_seen"] = datetime.now()
        self.tracks[track_id]["consecutive_misses"] = 0  # Reset consecutive misses counter

    def get_tracks(self):
        return self.tracks

    def increase_consecutive_misses(self, track_id):
        self.tracks[track_id]["consecutive_misses"] += 1


def process_video(video_path, output_json, process_id):
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True).autoshape()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = f'output_video_{process_id}.avi'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracker = Tracker()
    consecutive_misses_threshold = 5  # Number of consecutive misses before assigning a new tracking ID
    delta_threshold = 20  # Delta threshold for centroid matching

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        frame_with_boxes = results.render()[0]

        # Count the number of people detected
        num_people = 0

        # Initialize a dictionary to store centroids for each person
        centroids = {}

        for box in results.xyxy[0]:
            if box[-1] == 0:  # Assuming 0 corresponds to the class index for people
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                centroids[(x_center, y_center)] = box
                num_people += 1

        for centroid, box in centroids.items():
            # Draw bounding box
            cv2.rectangle(frame_with_boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # Update or add track based on the centroid
            track_id = None
            for existing_track_id, existing_track_info in tracker.get_tracks().items():
                existing_centroid = existing_track_info["centroid"]
                if abs(existing_centroid[0] - centroid[0]) <= delta_threshold and abs(existing_centroid[1] - centroid[1]) <= delta_threshold:
                    track_id = existing_track_id
                    tracker.increase_consecutive_misses(track_id)
                    break

            if not track_id or tracker.get_tracks()[track_id]["consecutive_misses"] >= consecutive_misses_threshold:
                track_id = tracker.add_track(centroid)

            # Display tracking ID and time
            current_time = datetime.now()
            duration = current_time - tracker.get_tracks()[track_id]["first_seen"]
            duration_str = str(duration).split('.')[0]  # Format duration as HH:MM:SS
            print(f"Process {process_id} - Track ID: {track_id}, Duration: {duration_str}")

            cv2.putText(frame_with_boxes, f'Cam {process_id} Track ID: {track_id}', (int(box[0]), int(box[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_with_boxes, f'Duration: {duration_str}', (int(box[0]), int(box[1]) + 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Update track with new centroid
            tracker.update_track(track_id, centroid)

        # Increase consecutive misses for tracks without detection
        for existing_track_id, existing_track_info in tracker.get_tracks().items():
            existing_centroid = existing_track_info["centroid"]
            if existing_centroid not in centroids:
                tracker.increase_consecutive_misses(existing_track_id)

        # Display the total number of people
        cv2.putText(frame_with_boxes, f'Total People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame_with_boxes)

    # Save tracking information to output JSON
    output_json.extend(tracker.get_tracks())

    cap.release()
    out.release()


if __name__ == '__main__':
    video_paths = ['/content/drive/MyDrive/Joshi/test_2.mp4', '/content/drive/MyDrive/Joshi/test_3.mp4', '/content/drive/MyDrive/Joshi/test_4.mp4']
    output_json = []

    with Manager() as manager:
        processes = []
        for i, video_path in enumerate(video_paths):
            process = Process(target=process_video, args=(video_path, output_json, i + 1))
            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        output_json_path = f'output_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
        with open(output_json_path, 'w') as json_file:
            json.dump(output_json, json_file, indent=4)

        print(f'Output JSON saved to {output_json_path}')
