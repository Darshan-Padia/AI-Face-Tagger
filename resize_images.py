import os
import cv2

def resize_images(input_folder, output_folder, target_size=(100, 100)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Walk through the input folder and its subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Check for image files
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Read and resize the image
                img = cv2.imread(input_path)
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Save the resized image
                cv2.imwrite(output_path, resized_img)
                print(f"Resized {input_path} to {output_path}")

# Main function
if __name__ == "__main__":
    input_folder = "cropped_faces"
    output_folder = "resized_faces"
    target_size = (100, 100)  # Set the desired size
    
    # Resize images
    resize_images(input_folder, output_folder, target_size)
