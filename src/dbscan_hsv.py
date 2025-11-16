import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


def resize_image(image, width, height):
    try:
        resized_image = cv2.resize(image, (width, height))
        print(f"Successfully resized image to {width}x{height}")
        return resized_image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None


def extract_features(image):
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bins = (8, 8, 8)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        print(f"Error extracting features from image: {e}")
        return None


def load_images_from_directory(directory):
    images = []
    image_names = []
    try:
        print(f"Checking directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        image_names.append(file)
                    else:
                        print(f"Failed to load {img_path}")
                else:
                    print(f"Unsupported file format: {file}")
        return images, image_names
    except Exception as e:
        print(f"Error loading images from directory: {e}")
        return images, image_names


def process_directory(directory_path, plant_name, target_size):
    try:
        plant_images, image_names = load_images_from_directory(directory_path)
        if not plant_images:
            print(f"No images found in directory: {directory_path}")
            return [], [], []
        plant_images_resized = [resize_image(plant_image, *target_size) for plant_image in plant_images]
        return plant_images_resized, [plant_name] * len(plant_images_resized), image_names
    except Exception as e:
        print(f"Error processing directory {directory_path}: {e}")
        return [], [], []


def main():
    try:
        with open('data/PlantNet300K/plant_names.json', 'r') as file:
            plant_names = json.load(file)
    except Exception as e:
        print(f"Failed to load plant names JSON: {e}")
        return

    base_directory_path = "data/PlantNet300K/train"
    target_size = (500, 500)

    all_features = defaultdict(list)
    plant_images_resized = []
    plant_labels = []
    plant_image_paths = []

    try:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_directory, os.path.join(base_directory_path, directory_index), plant_name, target_size)
                for directory_index, plant_name in plant_names.items()
            ]
            for future in futures:
                plant_images, labels, image_paths = future.result()
                for img, label, img_path in zip(plant_images, labels, image_paths):
                    features = extract_features(img)
                    if features is not None:
                        all_features[label].append((features, img, img_path))
                        plant_images_resized.append(img)
                        plant_labels.append(label)
                        plant_image_paths.append(img_path)

        if not all_features:
            print("No images found in any of the directories.")
            return

        user_image_path = "data/PlantNet300K/test/example_leaf.jpg"
        user_image = cv2.imread(user_image_path)
        if user_image is None:
            print(f"Failed to load user image: {user_image_path}")
            return

        user_image_resized = resize_image(user_image, *target_size)
        if user_image_resized is None:
            print("Failed to resize user image.")
            return

        user_features = extract_features(user_image_resized)
        if user_features is None:
            print("Failed to extract features from user image.")
            return

        user_image_label = "User Image"

        best_match = {}
        for label, features_list in all_features.items():
            min_distance = float('inf')
            best_img = None
            best_img_path = None
            for features, img, img_path in features_list:
                distance = np.linalg.norm(user_features - features)
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

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

