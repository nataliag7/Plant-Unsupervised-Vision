import os
import json
import cv2
import numpy as np
import scipy.cluster.hierarchy as sh
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image, width, height):
    try:
        return cv2.resize(image, (width, height))
    except cv2.error as e:
        logging.error(f"Error resizing image: {e}")
        return None

def extract_features_sift(image):
    try:
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(image, None)
        return descriptors
    except cv2.error as e:
        logging.error(f"Error extracting features: {e}")
        return None

def compare_images_sift(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    if not matches:
        return np.inf
    return np.mean([m.distance for m in matches])

def load_images_from_directory(directory):
    images = []
    image_names = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        image_names.append(file)
                    else:
                        logging.warning(f"Failed to load {img_path}")
        return images, image_names
    except Exception as e:
        logging.error(f"Error loading images from directory: {e}")
        return images, image_names

def process_directory(directory_path, plant_name, target_size):
    try:
        plant_images, image_names = load_images_from_directory(directory_path)
        if not plant_images:
            return [], []
        plant_images_resized = [resize_image(img, *target_size) for img in plant_images]
        return plant_images_resized, [plant_name] * len(plant_images_resized)
    except Exception as e:
        logging.error(f"Error processing directory {directory_path}: {e}")
        return [], []

def process_batch(directories, plant_names, target_size):
    all_features = defaultdict(list)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_directory, path, plant_names[os.path.basename(path)], target_size)
            for path in directories
        ]
        for future in as_completed(futures):
            plant_images_resized, labels = future.result()
            for img, label in zip(plant_images_resized, labels):
                descriptors = extract_features_sift(img)
                if descriptors is not None:
                    all_features[label].append(descriptors)
    return all_features

def hierarchical_clustering(all_features, user_descriptors, labels):
    num_images = len(all_features)
    distances = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i + 1, num_images):
            dist = np.mean([compare_images_sift(d1, d2) for d1 in all_features[i] for d2 in all_features[j]])
            distances[i, j] = dist
            distances[j, i] = dist

    condensed_distances = ssd.squareform(distances)
    linkage = sh.linkage(condensed_distances, method='ward')

    plt.figure(figsize=(20, 18))
    sh.dendrogram(linkage, labels=labels)
    plt.xlabel("Image")
    plt.ylabel("SIFT Feature Distance")
    plt.title("Hierarchical Clustering Dendrogram Including User Image")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

    user_image_index = labels.index("User Image")
    cluster_indices = sh.fcluster(linkage, t=4, criterion='maxclust')
    user_image_cluster = cluster_indices[user_image_index]

    same_cluster_indices = [i for i, x in enumerate(cluster_indices) if x == user_image_cluster]

    cluster_labels = [labels[i] for i in same_cluster_indices]
    cluster_distances = distances[np.ix_(same_cluster_indices, same_cluster_indices)]

    cluster_linkage = sh.linkage(ssd.squareform(cluster_distances), method='ward')

    plt.figure(figsize=(20, 18))
    sh.dendrogram(cluster_linkage, labels=cluster_labels)
    plt.xlabel("Image")
    plt.ylabel("SIFT Feature Distance")
    plt.title("Hierarchical Clustering Dendrogram (User Image Cluster)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

def main():
    try:
        with open('path/to/file', 'r') as file:
            plant_names = json.load(file)
    except Exception as e:
        logging.error(f"Failed to load plant names JSON: {e}")
        return

    base_directory_path = "path/to/file"
    target_size = (300, 300)
    all_features = defaultdict(list)

    directories = [os.path.join(base_directory_path, directory_index) for directory_index in plant_names.keys()]
    batch_size = 500
    batches = [directories[i:i + batch_size] for i in range(0, len(directories), batch_size)]

    try:
        for batch in batches:
            batch_features = process_batch(batch, plant_names, target_size)
            for label, descriptors_list in batch_features.items():
                all_features[label].extend(descriptors_list)

        if not all_features:
            logging.warning("No images found in any of the directories.")
            return

        user_image_path = "path/to/file"
        user_image = cv2.imread(user_image_path)
        if user_image is None:
            logging.error(f"Failed to load user image: {user_image_path}")
            return

        user_image_resized = resize_image(user_image, *target_size)
        if user_image_resized is None:
            logging.error("Failed to resize user image.")
            return

        user_descriptors = extract_features_sift(user_image_resized)
        if user_descriptors is None:
            logging.error("Failed to extract features from user image.")
            return

        labels = list(all_features.keys()) + ["User Image"]
        all_descriptors = list(all_features.values()) + [[user_descriptors]]

        hierarchical_clustering(all_descriptors, user_descriptors, labels)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
