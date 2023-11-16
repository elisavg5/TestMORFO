import csv
import numpy as np
import cv2
import pandas as pd

##1
def generate_random_images(batch_size, image_shape):
    return np.random.randint(0, 256, size=(batch_size, *image_shape), dtype=np.uint8)


##2 
def add_random_white_and_black_pixels(images, percentage):
    num_pixels_to_change = int(np.prod(images.shape[1:]) * (percentage / 100))
    for image in images:
        white_pixels = np.random.choice(np.prod(image.shape[:-1]), num_pixels_to_change, replace=False)
        black_pixels = np.random.choice(np.prod(image.shape[:-1]), num_pixels_to_change, replace=False)
        # Convert flattened indices to 2D indices
        white_pixels = np.unravel_index(white_pixels, image.shape[:-1])
        black_pixels = np.unravel_index(black_pixels, image.shape[:-1])
        # Set selected pixels to white and black
        image[white_pixels[0], white_pixels[1], :] = [255, 255, 255]
        image[black_pixels[0], black_pixels[1], :] = [0, 0, 0]
    return images

##3
def apply_gaussian_blur(images, kernel_size=(5, 5), sigma_x=0):
    return [cv2.GaussianBlur(image, kernel_size, sigma_x) for image in images]

##4 (not working)
def calculate_color_statistics(images, color):
    color_values = {
        'White': [255, 255, 255],
        'Black': [0, 0, 0]
    }
    flattened_images = images.reshape(images.shape[0], -1, images.shape[-1])
    color_pixels = np.sum(np.all(np.isclose(flattened_images, np.array(color_values[color]).reshape(1, 1, -1, images.shape[-1])), axis=-1, keepdims=True), axis=(1, 2))
    non_zero_indices = np.any(color_pixels != 0, axis=1)
    statistics = {
        'Average': np.mean(color_pixels[non_zero_indices]),
        'Min': np.min(color_pixels[non_zero_indices]),
        'Max': np.max(color_pixels[non_zero_indices]),
        'StdDev': np.std(color_pixels[non_zero_indices])
    }
    return statistics



##1
# Number of batches and images per batch
num_batches = 10
images_per_batch = 10

# Image shape
image_shape = (256, 512, 3)

# Generate and store batches in a list
image_batches = [generate_random_images(images_per_batch, image_shape) for _ in range(num_batches)]

##2
percentage_of_pixels_to_change = 1
image_batches_with_pixels = [add_random_white_and_black_pixels(images, percentage_of_pixels_to_change)
                             for images in image_batches]

##3
image_batches_with_blur = apply_gaussian_blur(image_batches_with_pixels)

##5
batch_statistics = []
##4
for batch_index, images in enumerate(image_batches_with_blur):
    # Apply the modifications to each batch
    calculate_color_statistics_white = calculate_color_statistics(images, 'White')
    calculate_color_statistics_black = calculate_color_statistics(images, 'Black')

    ##5
    # Append statistics to the list
    batch_statistics.append({
        'batch_id': batch_index + 1,
        'white_avg': calculate_color_statistics_white['Average'],
        'white_min': calculate_color_statistics_white['Min'],
        'white_max': calculate_color_statistics_white['Max'],
        'white_std': calculate_color_statistics_white['StdDev'],
        'black_avg': calculate_color_statistics_black['Average'],
        'black_min': calculate_color_statistics_black['Min'],
        'black_max': calculate_color_statistics_black['Max'],
        'black_std': calculate_color_statistics_black['StdDev']
    })

# Save batches to CSV files
for batch_index, images in enumerate(image_batches):
    csv_filename = f'batch_{batch_index + 1}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['R', 'G', 'B'])
        # Write image data
        for image in images:
            csv_writer.writerows(image.reshape(-1, 3))

# Load batches back into memory
loaded_batches = []
for batch_index in range(num_batches):
    csv_filename = f'batch_{batch_index + 1}.csv'
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip header
        next(csv_reader)
        # Read image data
        image_data = np.array([list(map(int, row)) for row in csv_reader])
        loaded_batches.append(image_data.reshape((images_per_batch,) + image_shape))

# Check the shape of the loaded batches
for batch_index, loaded_images in enumerate(loaded_batches):
    print(f"Shape of loaded batch {batch_index + 1}: {loaded_images.shape}")


##5
# Create a DataFrame from the list of batch statistics
df = pd.DataFrame(batch_statistics)

# Save the DataFrame to a CSV file
df.to_csv('batch_statistics.csv', index=False)


##6
# Load the CSV file into a DataFrame
df = pd.read_csv('batch_statistics.csv')

# Save the DataFrame to Parquet format
df.to_parquet('batch_statistics.parquet', index=False)

