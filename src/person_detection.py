import cv2
import torch
import json
from datetime import datetime

# Load YOLOv7 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True).autoshape()

# Input and output video paths
#input_video_path = 'test_1.mp4'
input_video_path= '/content/drive/MyDrive/Joshi/test_1.mp4'
output_video_path = 'output_video_2.avi'

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize variables to store information for JSON output
output_json = []
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Perform inference
    results = model(frame)
    #print(results.render())

    # Increment the frame count
    frame_count += 1

    # Extract relevant information for JSON output
    output_info = {
        "Frame-Count": frame_count,
        "output boxes": results.xyxy[0].tolist(),
        #"classes": results.names.strip().split('\n'),
        "timestamp": str(datetime.now()),
        "Device ID": "Cam-1"
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

    # Display the frame with bounding boxes (optional)
    # cv2.imshow('YOLOv7 Person Detection', frame_with_boxes)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save the output JSON to a file
output_json_path = 'output.json'
with open(output_json_path, 'w') as json_file:
    json.dump(output_json, json_file, indent=4)

print(f'Output JSON saved to {output_json_path}')
