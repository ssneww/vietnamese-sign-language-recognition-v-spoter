import torch
import os
from torch.utils.data import DataLoader
from spoter.spoter_model import SPOTER
from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.utils import evaluate
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def test_model_from_csv(testing_set_path, model_checkpoint_path, num_classes, hidden_dim, device='cpu'):
    """
    Function to test a trained model using data from a CSV file.

    Parameters:
    - testing_set_path (str): Path to the CSV file containing the test data.
    - model_checkpoint_path (str): Path to the model checkpoint (.pth file).
    - num_classes (int): Number of output classes of the model.
    - hidden_dim (int): Hidden dimension size of the model.
    - device (str): Compute device ('cpu' or 'cuda').

    Returns:
    - The label index that the model predicts with perfect accuracy (eval_acc == 1),
      or None if no perfect match is found.
    """
    
    # Set device: switch to CPU if CUDA is requested but unavailable
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    device = torch.device(device)

    # Create dataset from the CSV file
    dataset = CzechSLRDataset(testing_set_path, augmentations=False)

    # Try labels from 1 to 99 by temporarily modifying the first sample's label
    for i in range(1, 100):
        dataset.labels[0] = i  # Temporarily set the first sample's label to i
        eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Check if the checkpoint file exists
        if not os.path.exists(model_checkpoint_path):
            print(f"Checkpoint {model_checkpoint_path} does not exist.")
            return None

        # Load the model and checkpoint, then map to the chosen device
        tested_model = SPOTER(num_classes=num_classes, hidden_dim=hidden_dim)
        tested_model = torch.load(model_checkpoint_path, map_location=device)
        tested_model.to(device)
        tested_model.eval()  # Set model to evaluation mode

        # Evaluate model on the test data
        _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=False)

        # If accuracy is perfect, return the corresponding label index
        if eval_acc == 1:
            return i - 1

def put_vietnamese_text(img, text, position, font_size=32, color=(0, 0, 255)):
    """
    Draws Vietnamese text onto an OpenCV image.

    Parameters:
    - img (np.ndarray): The image in BGR format.
    - text (str): The text to draw.
    - position (tuple): (x, y) coordinates for text placement.
    - font_size (int): Font size.
    - color (tuple): BGR color for the text.
    
    Returns:
    - The image with text drawn.
    """
    # Convert the image to a PIL Image for proper font rendering
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Path to a font that supports Vietnamese characters
    font_path = "C:/Windows/Fonts/arial.ttf"  # Change to the full path if needed
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print("⚠️ 'arial.ttf' font not found, using default font (Vietnamese accents may not render correctly).")
        font = ImageFont.load_default()
    
    # Draw the text onto the image
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV BGR format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_video(video_path, label_text):
    """
    Processes a video by overlaying the predicted label text on each frame,
    then saves and optionally displays the result.

    Parameters:
    - video_path (str): Path to the input video file.
    - label_text (str): The text label to overlay.
    
    Returns:
    - Path to the processed output video file.
    """
    # Open the input video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video")
        return None

    # Retrieve video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare VideoWriter for the processed output
    out_path = os.path.splitext(video_path)[0] + "_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop when no more frames

        # Overlay the label text in the bottom-left corner
        frame = put_vietnamese_text(frame, label_text, (10, frame_height - 50), font_size=40, color=(0, 0, 255))

        # Write the processed frame to the output video
        out.write(frame)

        # Optionally display the processed frame
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit playback

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return out_path

if __name__ == "__main__":
    output_csv = sys.argv[1]
    video_path = sys.argv[2]
    label_index = test_model_from_csv(output_csv, "checkpoint_v_10.pth", num_classes=100, hidden_dim=108, device='cuda')
    
    label_videos = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "90", "ai cho", "ai", "anh chị", "anh em", "anh hai", 
        "anh họ", "bay", "bé trai", "béo", "bìa sách", "chia sẻ", "chào", "cháu ngoại", "cháu nội", "chính phủ",
        "công cụ", "công sức", "cổ vũ", "củ khoai tây", "cử động", "dự trữ", "giao hàng", "giấy", "hiện đại", "hoa cúc",
        "Hoàng tử", "hòn đá", "hói đầu", "hẹn", "hợp lí", "khách sạn", "khám bệnh", "khói", "mùa hè", "mục tiêu", 
        "người yêu", "nhược điểm", "năn nỉ", "quả lê", "quả mít", "quả táo", "quả vải", "quả địa cầu", "rửa mặt",
        "son môi", "sân nhà", "tham khảo", "thuỷ thủ", "thông tin", "thủ công", "thực tế", "tia nắng", "tiêu cực", 
        "tiến sĩ", "truyền thông", "trường tiểu học", "trứng", "tái phát", "tôn thờ", "tải về", "tỉ số", "tồn tại", 
        "xa xôi", "ác cảm", "ác", "ân hận", "ôm", "ôn luyện", "ăn cóc", "ăn diện", "ăn hối lộ", "ăn vú sữa (bằng thìa)", 
        "ăn vú sữa", "ăn vặt", "ăn vụng", "ăn vừa", "ăn xin", "ăn ít", "điều khiển", "đùm bọc", "đại tướng", "định vị", 
        "ấm no", "ấm áp", "ấm đun nước", "ấn tượng", "ấn", "ốm đau"
    ]
    label_text = label_videos[label_index]
    processed_video_file = process_video(video_path, label_text)
    subprocess.run(['start', processed_video_file], shell=True)
