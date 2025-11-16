import os
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))


def compare_images_mse(user_image, plant_image):
    return np.mean((user_image - plant_image) ** 2)


def compare_images_ssim(user_image, plant_image):
    height, width, _ = user_image.shape
    win_size = min(height, width) // 2
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)
    return ssim(user_image, plant_image, multichannel=True, win_size=win_size, channel_axis=2)


def load_images_from_directory(directory):
    images = []
    image_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                img_path = os.path.join(root, file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        image_names.append(file)
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")
    return images, image_names


def main():
    base_directory_path = "data/PlantNet300K/train"

    try:
        user_image_path = "data/PlantNet300K/test/example_leaf.jpg"
        user_image = cv2.imread(user_image_path)
        if user_image is None:
            print(f"Failed to load user image: {user_image_path}")
            return

        target_width, target_height = 500, 500
        user_image_resized = resize_image(user_image, target_width, target_height)

        overall_best_match_mse = float('inf')
        overall_best_match_ssim = -float('inf')

        overall_best_plant_name_mse = ""
        overall_best_plant_name_ssim = ""

        with open('data/PlantNet300K/plant_names.json', 'r') as file:
            plant_names = json.load(file)

        for directory_index, plant_name in plant_names.items():
            directory_path = os.path.join(base_directory_path, directory_index)
            plant_images, _ = load_images_from_directory(directory_path)

            if not plant_images:
                continue

            plant_images_resized = [resize_image(plant_image, target_width, target_height) for plant_image in plant_images]

            mse_similarities = [compare_images_mse(user_image_resized, plant_image) for plant_image in plant_images_resized]
            ssim_similarities = [compare_images_ssim(user_image_resized, plant_image) for plant_image in plant_images_resized]

            best_match_index_mse = np.argmin(mse_similarities)
            best_match_similarity_mse = mse_similarities[best_match_index_mse]

            best_match_index_ssim = np.argmax(ssim_similarities)
            best_match_similarity_ssim = ssim_similarities[best_match_index_ssim]


            if best_match_similarity_mse < overall_best_match_mse:
                overall_best_match_mse = best_match_similarity_mse
                overall_best_plant_name_mse = plant_name

            if best_match_similarity_ssim > overall_best_match_ssim:
                overall_best_match_ssim = best_match_similarity_ssim
                overall_best_plant_name_ssim = plant_name


        print("MSE-based results:")
        print(f"The most similar plant image (MSE) has similarity: {overall_best_match_mse}")
        print(f"Plant name: {overall_best_plant_name_mse}")

        print("SSIM-based results:")
        print(f"The most similar plant image (SSIM) has similarity: {overall_best_match_ssim}")
        print(f"Plant name: {overall_best_plant_name_ssim}")


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()


