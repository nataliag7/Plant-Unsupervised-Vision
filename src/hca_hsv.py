import os
import json
import cv2
import numpy as np
import scipy.cluster.hierarchy as sh
import scipy.spatial.distance as ssd
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

def compare_images_euclidean(features1, features2):
    if features1 is None or features2 is None:
        return float('inf')
    distance = np.linalg.norm(features1 - features2)
    return distance

def load_images_from_directory(directory):
    images = []
    image_names = []
    try:
        print(f"Checking directory: {directory}")
        for root, _, files in os.walk(directory):
            print(f"Files in directory '{root}': {files}")
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
            return [], []
        plant_images_resized = [resize_image(plant_image, *target_size) for plant_image in plant_images]
        return plant_images_resized, [plant_name] * len(plant_images_resized)
    except Exception as e:
        print(f"Error processing directory {directory_path}: {e}")
        return [], []

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

    try:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_directory, os.path.join(base_directory_path, directory_index), plant_name, target_size)
                for directory_index, plant_name in plant_names.items()
            ]
            for future in futures:
                plant_images_resized, labels = future.result()
                for img, label in zip(plant_images_resized, labels):
                    features = extract_features(img)
                    if features is not None:
                        all_features[label].append(features)

        if not all_features:
            print("No images found in any of the directories.")
            return

        # Load and process the user image
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

        averaged_features = {label: np.mean(np.array(features), axis=0) for label, features in all_features.items()}

        averaged_features[user_image_label] = user_features

        labels = list(averaged_features.keys())
        feature_matrix = np.array(list(averaged_features.values()))

        num_images = len(feature_matrix)
        distances = np.zeros((num_images, num_images))

        for i in range(num_images):
            for j in range(i + 1, num_images):
                distances[i, j] = compare_images_euclidean(feature_matrix[i], feature_matrix[j])
                distances[j, i] = distances[i, j]

        condensed_distances = ssd.squareform(distances)
        linkage = sh.linkage(condensed_distances, method='ward')

        plt.figure(figsize=(20, 18))
        dendrogram = sh.dendrogram(linkage, labels=labels)
        plt.xlabel("Image", fontsize=8)
        plt.ylabel("Euclidean Distance", fontsize=8)
        plt.xticks(rotation=90, fontsize=4)
        plt.yticks(fontsize=8)
        plt.title("Hierarchical Clustering Dendrogram Including User Image")
        plt.show()

        user_image_index = labels.index(user_image_label)
        cluster_indices = sh.fcluster(linkage, t=4, criterion='maxclust')
        user_image_cluster = cluster_indices[user_image_index]

        same_cluster_indices = [i for i, x in enumerate(cluster_indices) if x == user_image_cluster]

        cluster_labels = [labels[i] for i in same_cluster_indices]
        cluster_distances = ssd.squareform(condensed_distances)[np.ix_(same_cluster_indices, same_cluster_indices)]

        cluster_linkage = sh.linkage(ssd.squareform(cluster_distances), method='ward')

        plt.figure(figsize=(20, 18))
        cluster_dendrogram = sh.dendrogram(cluster_linkage, labels=cluster_labels)
        plt.xlabel("Image", fontsize=8)
        plt.ylabel("Euclidean Distance", fontsize=8)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.title("Hierarchical Clustering Dendrogram (User Image Cluster)")
        plt.show()

    except Exception as e:
        print(f"An error occurred during the main process: {e}")

if __name__ == "__main__":
    main()

