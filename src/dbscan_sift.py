import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def resize_image(image, width, height):
    try:
        return cv2.resize(image, (width, height))
    except cv2.error as e:
        print(f"Error resizing image: {e}")
        return None

def extract_features_sift(image):
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors
    except cv2.error as e:
        print(f"Error extracting features: {e}")
        return None

def compare_images_sift(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return np.inf
    if descriptors1.dtype != descriptors2.dtype:
        descriptors1 = descriptors1.astype(np.float32)
        descriptors2 = descriptors2.astype(np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return np.mean([m.distance for m in matches]) if matches else np.inf

def load_images_from_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    yield img, file
                else:
                    print(f"Failed to load {img_path}")

def process_directory(directory_path, plant_name, target_size):
    plant_images_resized = []
    image_names = []
    for img, file in load_images_from_directory(directory_path):
        resized_img = resize_image(img, *target_size)
        if resized_img is not None:
            plant_images_resized.append(resized_img)
            image_names.append(file)
    return plant_images_resized, [plant_name] * len(plant_images_resized), image_names

def process_batch(directories, plant_names, target_size):
    all_features = defaultdict(list)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_directory, path, plant_names[os.path.basename(path)], target_size)
            for path in directories
        ]
        for future in as_completed(futures):
            try:
                plant_images, labels, image_paths = future.result()
                for img, label, img_path in zip(plant_images, labels, image_paths):
                    features = extract_features_sift(img)
                    if features is not None:
                        all_features[label].append((features, img, img_path))
            except Exception as e:
                print(f"Error processing directory: {e}")
    return all_features

def main():
    try:
        with open('path/to/file', 'r') as file:
            plant_names = json.load(file)
    except Exception as e:
        print(f"Failed to load plant names JSON: {e}")
        return

    base_directory_path = "path/to/file"
    target_size = (300, 300)

    directories = [os.path.join(base_directory_path, directory_index) for directory_index in plant_names.keys()]
    batch_size = 100
    batches = [directories[i:i + batch_size] for i in range(0, len(directories), batch_size)]

    all_features = defaultdict(list)

    for batch in batches:
        batch_features = process_batch(batch, plant_names, target_size)
        for label, descriptors_list in batch_features.items():
            all_features[label].extend(descriptors_list)

    if not all_features:
        print("No images found in any of the directories.")
        return

    user_image_path = "path/to/file"
    user_image = cv2.imread(user_image_path)
    if user_image is None:
        print(f"Failed to load user image: {user_image_path}")
        return

    user_image_resized = resize_image(user_image, *target_size)
    if user_image_resized is None:
        print("Failed to resize user image.")
        return

    user_features = extract_features_sift(user_image_resized)
    if user_features is None:
        print("Failed to extract features from user image.")
        return

    best_match = {}
    for label, features_list in all_features.items():
        min_distance = float('inf')
        best_img = None
        best_img_path = None
        for features, img, img_path in features_list:
            distance = compare_images_sift(user_features, features)
            if distance < min_distance:
                min_distance = distance
                best_img = img
                best_img_path = img_path
        best_match[label] = (min_distance, best_img, best_img_path)

    sorted_matches = sorted(best_match.items(), key=lambda x: x[1][0])
    top_matches = sorted_matches[:5]

    print("Top matches:")
    for label, (distance, _, img_path) in top_matches:
        print(f"Plant: {label}, Distance: {distance}, Image Path: {img_path}")

    plt.figure(figsize=(15, 6))
    plt.subplot(2, len(top_matches) + 1, 1)
    plt.imshow(cv2.cvtColor(user_image_resized, cv2.COLOR_BGR2RGB))
    plt.title("User Image")
    plt.axis('off')

    for i, (label, (distance, img, _)) in enumerate(top_matches):
        plt.subplot(2, len(top_matches) + 1, i + 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{label}: {distance:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
