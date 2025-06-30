import cv2
import mediapipe as mp
import pandas as pd
import os
from multiprocessing import Pool

# Khởi tạo MediaPipe Hands và Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Đường dẫn tới thư mục train và tên file CSV đầu ra
train_dir = r"test2"  # Đổi thành đường dẫn tới thư mục train của bạn
output_csv = "test2.csv"

# Danh sách tên các điểm landmarks cho cả pose và hands
pose_landmark_names = [
    "nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar", 
    "rightShoulder", "leftShoulder", "rightElbow", "leftElbow", 
    "rightWrist", "leftWrist"
]

hand_landmark_names = [
    "wrist", "thumbCMC", "thumbMP", "thumbIP", "thumbTip",
    "indexMCP", "indexPIP", "indexDIP", "indexTip",
    "middleMCP", "middlePIP", "middleDIP", "middleTip",
    "ringMCP", "ringPIP", "ringDIP", "ringTip",
    "littleMCP", "littlePIP", "littleDIP",  "littleTip"
]

# Tạo tên cột cho các điểm pose và hand (cho cả hai tay), và thêm cột label cùng với các thông tin video
columns = ["labels", "video_fps", "video_size_width", "video_size_height"]
for name in pose_landmark_names:
    columns.append(f"{name}_X")
    columns.append(f"{name}_Y")

for hand in ["left", "right"]:
    for name in hand_landmark_names:
        columns.append(f"{name}_{hand}_X")
        columns.append(f"{name}_{hand}_Y")

# Hàm xử lý video
def process_video(video_path, folder_name, columns):
    # Mở video để lấy fps và kích thước khung hình
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_size_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo dictionary để lưu danh sách tọa độ cho từng video
    video_data = {col: [] for col in columns}
    video_data["labels"] = folder_name  # Gán nhãn (label) cho video
    video_data["video_fps"] = video_fps
    video_data["video_size_width"] = video_size_width
    video_data["video_size_height"] = video_size_height

    # Xử lý từng khung hình trong video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi khung hình sang RGB (MediaPipe yêu cầu đầu vào là RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Xử lý khung hình bằng MediaPipe Pose và Hands
        pose_results = pose.process(image_rgb)
        hands_results = hands.process(image_rgb)

        # Khởi tạo danh sách với giá trị 0 cho mỗi điểm pose (12 điểm * 2 tọa độ) và mỗi điểm hand (42 điểm * 2 tọa độ)
        frame_data = [0] * (len(columns) - 4)  # Trừ 4 để không tính các cột "labels", "video_fps", "video_size_width", "video_size_height"

        # Nếu phát hiện các điểm pose trong khung hình
        if pose_results.pose_landmarks:
            # Lấy tọa độ cho từng điểm pose cụ thể
            pose_landmarks = pose_results.pose_landmarks.landmark

            # Chỉ số cho các điểm pose trong mô hình MediaPipe Pose
            pose_indices = {
                "nose": 0, "rightEye": 5, "leftEye": 2, "rightEar": 8, "leftEar": 7,
                "rightShoulder": 12, "leftShoulder": 11, "rightElbow": 14,
                "leftElbow": 13, "rightWrist": 16, "leftWrist": 15
            }

            # Lấy tọa độ và lưu vào frame_data cho từng điểm pose
            for name, idx in pose_indices.items():
                frame_data[columns.index(f"{name}_X") - 4] = pose_landmarks[idx].x
                frame_data[columns.index(f"{name}_Y") - 4] = pose_landmarks[idx].y

            # Tính trung điểm của hai vai
            right_shoulder = pose_landmarks[pose_indices["rightShoulder"]]
            left_shoulder = pose_landmarks[pose_indices["leftShoulder"]]
            shoulders_mid_x = (right_shoulder.x + left_shoulder.x) / 2
            shoulders_mid_y = (right_shoulder.y + left_shoulder.y) / 2

            # Tính "neck" bằng cách lấy trung điểm giữa "nose" và trung điểm của hai vai
            nose = pose_landmarks[pose_indices["nose"]]
            neck_x = (nose.x + shoulders_mid_x) / 2
            neck_y = (nose.y + shoulders_mid_y) / 2
            frame_data[columns.index("neck_X") - 4] = neck_x
            frame_data[columns.index("neck_Y") - 4] = neck_y

        # Nếu phát hiện các điểm hand trong khung hình
        if hands_results.multi_hand_landmarks:
            # Dữ liệu cho hai tay, mỗi tay có 21 điểm (mỗi điểm có tọa độ X và Y)
            left_hand_data = [0] * (21 * 2)
            right_hand_data = [0] * (21 * 2)

            for hand_index, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                # Lấy tọa độ cho từng điểm hand và lưu vào danh sách left_hand_data hoặc right_hand_data
                if hand_index == 0:  # Tay thứ nhất (giả định là tay trái)
                    left_hand_data = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y)]
                elif hand_index == 1:  # Tay thứ hai (giả định là tay phải)
                    right_hand_data = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y)]

            # Thêm tọa độ của từng điểm hand vào frame_data
            for i, col in enumerate(columns[len(pose_landmark_names)*2 + 4 : len(pose_landmark_names)*2 + 4 + 21*2]):
                frame_data[columns.index(col) - 4] = left_hand_data[i]

            for i, col in enumerate(columns[len(pose_landmark_names)*2 + 4 + 21*2:]):
                frame_data[columns.index(col) - 4] = right_hand_data[i]

        # Thêm mỗi tọa độ vào danh sách tương ứng trong video_data
        for i, col in enumerate(columns[4:]):
            video_data[col].append(frame_data[i])

    # Giải phóng tài nguyên video
    cap.release()

    return video_data


# Hàm xử lý tất cả video trong một thư mục
def process_folder(folder_name):
    folder_path = os.path.join(train_dir, folder_name)
    video_data_list = []
    
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        if os.path.isfile(video_path) and video_path.endswith((".mp4", ".avi", ".mov")):
            video_data = process_video(video_path, folder_name, columns)
            video_data_list.append(video_data)

    return video_data_list


# Dùng multiprocessing để xử lý song song
def process_all_videos():
    folder_names = os.listdir(train_dir)
    
    # Tạo Pool để chạy song song
    with Pool() as pool:
        all_video_data = pool.map(process_folder, folder_names)

    # Nối tất cả kết quả từ các thư mục vào một list chung
    all_video_data = [item for sublist in all_video_data for item in sublist]

    # Giải phóng tài nguyên MediaPipe
    hands.close()
    pose.close()

    # Chuyển tất cả dữ liệu video thành DataFrame và lưu vào CSV
    df = pd.DataFrame(all_video_data)
    df.to_csv(output_csv, index=False)

    print(f"Dữ liệu landmark cho toàn bộ video (cả pose và hand) đã được lưu vào {output_csv}")


if __name__ == "__main__":
    process_all_videos()
