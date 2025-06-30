import pandas as pd
import ast

def expand_middle_frames_alternate(frame_data):
    """
    Expand frames by alternately repeating the middle third of frames.
    """
    current_frames = len(frame_data)

    if current_frames == 0:
        return frame_data  # Return original data if there are no frames

    start_idx = current_frames // 3
    end_idx = 2 * (current_frames // 3)

    before_middle = frame_data[:start_idx]
    middle_frames = frame_data[start_idx:end_idx]
    after_middle = frame_data[end_idx:]

    expanded_middle = []
    for frame in middle_frames:
        expanded_middle.extend([frame, frame])  # Repeat each middle frame twice

    expanded_frames = before_middle + expanded_middle + after_middle

    return expanded_frames

def double_after_every_two_frames(frame_data):
    """
    Duplicate each frame after every two frames.
    """
    if len(frame_data) == 0:
        return frame_data

    expanded_frames = []
    for i in range(len(frame_data)):
        expanded_frames.append(frame_data[i])
        if (i + 1) % 2 == 0:
            expanded_frames.append(frame_data[i])  # Add the same frame again

    return expanded_frames

def expand_and_save_triple(csv_file, output_csv):
    """
    Expand frames in a CSV file and save to a new file, tripling the number of rows.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Identify landmark data columns (excluding static columns)
    static_columns = ["labels", "video_fps", "video_size_width", "video_size_height"]
    landmark_columns = [col for col in df.columns if col not in static_columns]

    # Create a new DataFrame to hold the expanded results
    expanded_df = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        original_row = row.copy()
        middle_expanded_row = row.copy()
        double_expanded_row = row.copy()

        for col in landmark_columns:
            frame_data = row[col]

            if isinstance(frame_data, str):
                try:
                    # Parse the string representation of the list
                    frame_data = ast.literal_eval(frame_data)
                except (ValueError, SyntaxError):
                    print(f"Could not parse data in column '{col}', row {index}. Skipping.")
                    continue

            if isinstance(frame_data, list):
                # Original data
                original_row[col] = str(frame_data)
                # Expand using the middle-frames method
                middle_expanded_row[col] = str(expand_middle_frames_alternate(frame_data))
                # Expand using the double-after-two-frames method
                double_expanded_row[col] = str(double_after_every_two_frames(frame_data))

        # Append the three versions (original + two expansions) to the expanded DataFrame
        expanded_df = pd.concat(
            [expanded_df, pd.DataFrame([original_row, middle_expanded_row, double_expanded_row])],
            ignore_index=True
        )

    # Save the expanded DataFrame to a new CSV file
    expanded_df.to_csv(output_csv, index=False)
    print(f"Expanded file has been saved at: {output_csv}")

# Example usage
if __name__ == "__main__":
    input_csv = "data/train_augment.csv"       # Path to the input CSV file
    output_csv = "data/train_augment_frame.csv"  # Path to the output CSV file

    # Run the expansion function
    expand_and_save_triple(input_csv, output_csv)
