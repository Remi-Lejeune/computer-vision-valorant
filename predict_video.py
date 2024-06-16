from ultralytics import YOLO
from ultralytics import RTDETR
import random
import cv2
import torch
import time


if __name__ == '__main__':
    random.seed(10)
    # print(f'CUDA version: {torch.version.cuda}')
    # cuda_id = torch.cuda.current_device()
    # print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    # print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
    # print(f'Torch version: {torch.__version__}')



    # choice = input("1=yolo, 2=fastsam, 3=rtdetr")
    choice = '3'
    if choice == '1':
        chosen_model = "weights/best_weights_yolo_final.pt"
        model = YOLO(chosen_model)
    elif choice == '2':
        chosen_model = "weights/best_weights_fastsam_final.pt"
        model = YOLO(chosen_model)
    else:
        chosen_model = "weights/best_weights_rtdetr_final.pt"
        model = RTDETR(chosen_model)


    video_path = "arj_cv_vid.mp4"
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    inference_times = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames

            start_time = time.time()

            results = model.track(frame, persist=True, conf=0.5)

            end_time = time.time()

            inference_time = end_time - start_time
            inference_times.append(inference_time)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            output_video.write(annotated_frame)
            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    print(f'Average inference time: {avg_inference_time:.4f} seconds')