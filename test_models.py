import numpy
from ultralytics import YOLO
from ultralytics import RTDETR
import random
import torch
from ultralytics.utils.benchmarks import benchmark

def calc_f1(precision, recall):
    return 2 * (recall * precision) / (recall + precision)

if __name__ == '__main__':
    random.seed(10)
    # print(calc_f1(0.881, 0.565))
    # print(f'CUDA version: {torch.version.cuda}')
    # cuda_id = torch.cuda.current_device()
    # print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    # print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
    # print(f'Torch version: {torch.__version__}')
    choice = input("1=yolo, 2=fastsam, 3=rtdetr")
    data = "C:/Users/Justin/Desktop/CV_DL/data/Santyasa.v4-tester-3.yolov9/data.yaml"
    if choice == '1':
        chosen_model = "weights/best_weights_yolo_final.pt"
        model = YOLO(chosen_model)
    elif choice == '2':
        chosen_model = "weights/best_weights_fastsam_final.pt"
        data = "C:/Users/Justin/Desktop/CV_DL/data/yolo_seg_data/cv/data.yaml"
        model = YOLO(chosen_model)
    else:
        chosen_model = "weights/best_weights_rtdetr_final.pt"
        model = RTDETR(chosen_model)
    choice = input("edge(=1) or not(=2)")
    if choice == '1':
        if chosen_model == 'weights/best_weights_fastsam_final.pt':
            data = "C:/Users/Justin/Desktop/CV_DL/data/edge_cases_segments/data.yaml"
        else:
            data = "C:/Users/Justin/Desktop/CV_DL/data/edge_cases/data.yaml"

    # metrics = model.val(conf=0.5, data=data, split="test")

    video_path = "arj_cv_vid.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.65)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

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
    cv2.destroyAllWindows()
