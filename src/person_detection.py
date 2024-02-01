import cv2
import torch
import json
from datetime import datetime
from multiprocessing import Process, Manager

def process_video(video_path, output_json, process_id):
    # Load YOLOv7 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True).autoshape()

    # Open the input video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = f'output_video_{process_id}.avi'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize variables to store information for JSON output
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Increment the frame count
        frame_count += 1

        # Extract relevant information for JSON output
        output_info = {
            "Frame-Count": frame_count,
            "output boxes": results.xyxy[0].tolist(),
            "timestamp": str(datetime.now()),
            "Device ID": f"Cam-{process_id}"
        }
        print(output_info)

        # Append the information to the output JSON list
        output_json.append(output_info)

        # Draw bounding boxes on the frame
        frame_with_boxes = results.render()[0]

        # Display the total number of people using cv2.putText
        num_people = (results.pred[0][:, -1] == 0).sum().item()
        cv2.putText(frame_with_boxes, f'Total People: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame with bounding boxes to the output video
        out.write(frame_with_boxes)

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__':
    # Input video paths
    video_paths = ['/content/drive/MyDrive/Joshi/test_1.mp4', '/content/drive/MyDrive/Joshi/test_1.mp4', '/content/drive/MyDrive/Joshi/test_1.mp4']

    # Create a manager to share data between processes
    with Manager() as manager:
        # Shared list for output JSON
        output_json = manager.list()

        # Create processes for each video feed
        processes = []
        for i, video_path in enumerate(video_paths):
            process = Process(target=process_video, args=(video_path, output_json, i+1))
            processes.append(process)

        # Start all processes
        for process in processes:
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Save the combined output JSON to a file
        output_json_path = 'output.json'
        with open(output_json_path, 'w') as json_file:
            json.dump(list(output_json), json_file, indent=4)

        print(f'Output JSON saved to {output_json_path}')

