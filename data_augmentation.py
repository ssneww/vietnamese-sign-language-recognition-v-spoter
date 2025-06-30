import pandas as pd
import random
import numpy as np
import ast
import torch
from augmentations import augment_rotate, augment_shear, augment_arm_joint_rotate, augment_rotate_hands_around_wrist
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

# Define HAND_IDENTIFIERS for both hands
HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]

def load_dataset(file_location: str):
    """
    Read a CSV file and convert it into depth maps for augmentation.
    """
    df = pd.read_csv(file_location, encoding="utf-8")
    # Rename columns: left→_0_, right→_1_
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    # Ensure neck coordinates exist
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    data = []
    labels = df["labels"].to_list()

    # Build depth map arrays
    for _, row in df.iterrows():
        current_row = np.empty(shape=(len(ast.literal_eval(row["leftEar_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)

    return data, df

def depth_map_to_dictionary(landmarks_tensor: np.ndarray) -> dict:
    """
    Convert a depth map from a NumPy array into a dictionary for augmentation.
    """
    output = {}
    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = landmarks_tensor[:, landmark_index].tolist()
    return output

def dictionary_to_depth_map(landmarks_dict: dict) -> np.ndarray:
    """
    Convert a dictionary of landmarks back into a NumPy depth map array.
    """
    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return output

def augment_depth_map(data, labels, video_fps, video_size_width, video_size_height, augmentations_prob=1.0, num_augmentations=1):
    """
    Apply all augmentations to the depth maps and return the augmented data along with labels and video info.
    
    :param data: list of depth maps
    :param labels: corresponding labels
    :param video_fps: list of video FPS values
    :param video_size_width: list of video widths
    :param video_size_height: list of video heights
    :param augmentations_prob: probability to apply each augmentation
    :param num_augmentations: number of times to apply augmentations (each augmentation type will be attempted)
    :return: augmented data, labels, and matching video info
    """
    augmented_data = []
    augmented_labels = []
    augmented_video_fps = []
    augmented_video_size_width = []
    augmented_video_size_height = []

    for depth_map, label, fps, width, height in zip(data, labels, video_fps, video_size_width, video_size_height):
        # For each augmentation iteration, copy the original depth map
        for _ in range(num_augmentations):
            depth_map_dict = depth_map_to_dictionary(depth_map)
            threshold = augmentations_prob

            # 1. Rotate
            if threshold > random.random():
                rotated = augment_rotate(depth_map_dict.copy(), (-13, 13))
                augmented_data.append(dictionary_to_depth_map(rotated))
                augmented_labels.append(label)
                augmented_video_fps.append(fps)
                augmented_video_size_width.append(width)
                augmented_video_size_height.append(height)

            # 2. Perspective shear
            if threshold > random.random():
                shear_persp = augment_shear(depth_map_dict.copy(), "perspective", (0, 0.1))
                augmented_data.append(dictionary_to_depth_map(shear_persp))
                augmented_labels.append(label)
                augmented_video_fps.append(fps)
                augmented_video_size_width.append(width)
                augmented_video_size_height.append(height)

            # 3. Squeeze shear
            if threshold > random.random():
                shear_sq = augment_shear(depth_map_dict.copy(), "squeeze", (0, 0.15))
                augmented_data.append(dictionary_to_depth_map(shear_sq))
                augmented_labels.append(label)
                augmented_video_fps.append(fps)
                augmented_video_size_width.append(width)
                augmented_video_size_height.append(height)

            # 4. Arm joint rotation
            if threshold > random.random():
                arm_rot = augment_arm_joint_rotate(depth_map_dict.copy(), 0.3, (-4, 4))
                augmented_data.append(dictionary_to_depth_map(arm_rot))
                augmented_labels.append(label)
                augmented_video_fps.append(fps)
                augmented_video_size_width.append(width)
                augmented_video_size_height.append(height)

            # 5. Rotate hands around the wrist (several angles)
            for i in range(5):
                if threshold > random.random():
                    hand_rot_p = augment_rotate_hands_around_wrist(depth_map_dict.copy(), (i+1)*3)
                    augmented_data.append(dictionary_to_depth_map(hand_rot_p))
                    augmented_labels.append(label)
                    augmented_video_fps.append(fps)
                    augmented_video_size_width.append(width)
                    augmented_video_size_height.append(height)
            for i in range(5):
                if threshold > random.random():
                    hand_rot_n = augment_rotate_hands_around_wrist(depth_map_dict.copy(), (i-1)*3)
                    augmented_data.append(dictionary_to_depth_map(hand_rot_n))
                    augmented_labels.append(label)
                    augmented_video_fps.append(fps)
                    augmented_video_size_width.append(width)
                    augmented_video_size_height.append(height)

    return augmented_data, augmented_labels, augmented_video_fps, augmented_video_size_width, augmented_video_size_height

def save_augmented_data_to_csv(input_csv: str, output_csv: str, augmentations_prob=0.5, num_augmentations=3):
    """
    Save the augmented depth map data to a new CSV, appending new rows to expand the dataset.
    
    :param input_csv: path to the original CSV
    :param output_csv: path for the augmented output CSV
    :param augmentations_prob: probability to apply each augmentation
    :param num_augmentations: number of augmentations per original sample
    """
    # Load original data
    data, df = load_dataset(input_csv)
    
    # Extract labels and unchanged video info columns
    labels = df["labels"].tolist()
    video_fps = df["video_fps"].tolist()
    video_size_width = df["video_size_width"].tolist()
    video_size_height = df["video_size_height"].tolist()

    # Perform augmentations
    augmented_data, augmented_labels, augmented_video_fps, augmented_video_size_width, augmented_video_size_height = augment_depth_map(
        data, labels, video_fps, video_size_width, video_size_height, augmentations_prob, num_augmentations
    )

    new_rows = []

    # Iterate over each augmented depth map
    for i, depth_map in enumerate(augmented_data):
        row = {}
        for idx, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            # Remove spaces for CSV storage
            row[identifier + "_X"] = str([frame[0] for frame in depth_map[:, idx, :]]).replace(' ', '')
            row[identifier + "_Y"] = str([frame[1] for frame in depth_map[:, idx, :]]).replace(' ', '')

        # Assign corresponding labels and video attributes
        row["labels"] = augmented_labels[i]
        row["video_fps"] = augmented_video_fps[i]
        row["video_size_width"] = augmented_video_size_width[i]
        row["video_size_height"] = augmented_video_size_height[i]

        new_rows.append(row)

    # Create a DataFrame for the augmented rows
    augmented_df = pd.DataFrame(new_rows)

    # Combine original and augmented data
    combined_df = pd.concat([df, augmented_df], ignore_index=True)

    # Rename columns back to left/right notation
    combined_df.columns = [item.replace("_0_", "_left_").replace("_1_", "_right_") for item in combined_df.columns]

    combined_df.to_csv(output_csv, index=False)
    print(f"Augmented depth map data has been saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_csv = "data/train.csv"
    output_csv = "data/train_augment.csv"
    save_augmented_data_to_csv(input_csv, output_csv, augmentations_prob=1.0, num_augmentations=1)
