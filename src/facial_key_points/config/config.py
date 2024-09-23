configuration = {
    "batch_size": 16,
    "model_input_size": (224, 224),
    "n_epochs": 10,
    "learning_rate": 0.001,
    "saved_path": "version2",
    "train_data_csv_path": "/home/roman/Facial_key_points/data/training_frames_keypoints.csv",
    "test_data_csv_path": "/home/roman/Facial_key_points/data/test_frames_keypoints.csv",
    "description": " Changed loss to nn.SmoothL1Loss",
}
