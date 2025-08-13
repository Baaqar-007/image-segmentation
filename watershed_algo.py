import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

def create_sample_image(path='circles.png'):
    """
    Generates a sample image with overlapping circles.
    This serves as the input for the watershed algorithm.
    The image is saved to disk.
    """
    # Create a black image
    image = np.zeros((400, 400), dtype=np.uint8)

    # Draw some white circles
    cv2.circle(image, (100, 100), 70, 255, -1)
    cv2.circle(image, (220, 150), 80, 255, -1)
    cv2.circle(image, (300, 280), 60, 255, -1)
    cv2.circle(image, (150, 280), 50, 255, -1)
    
    # Add some noise
    noise = np.zeros((400, 400), dtype=np.uint8)
    cv2.randu(noise, 0, 50)
    image = cv2.add(image, noise)

    cv2.imwrite(path, image)
    print(f"Sample image saved to '{path}'")
    return image

# --- Section 1: Image Loading and Pre-processing ---
def load_and_preprocess_image(image_path):
    """
    Loads an image from the specified path and performs initial pre-processing.
    Steps include:
    1. Loading the image (and converting to grayscale if necessary).
    2. Applying Otsu's thresholding to create a binary image.
    3. Using morphological opening to remove noise.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    # This automatically determines the optimal threshold value
    _ , thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return gray, opening

# --- Section 2: Preparation for Watershed ---
def prepare_for_watershed(opened_image):
    """
    Prepares the necessary components for the watershed algorithm.
    Steps include:
    1. Finding the sure background area using dilation.
    2. Finding the sure foreground (markers) using distance transform and thresholding.
    3. Identifying the unknown region between foreground and background.
    4. Creating initial labels for the markers.
    """
    # Sure background area
    sure_bg = cv2.dilate(opened_image, np.ones((3, 3), np.uint8), iterations=3)

    # Distance transform to find sure foreground
    dist_transform = cv2.distanceTransform(opened_image, cv2.DIST_L2, 5)
    
    # Threshold the distance transform to get the seeds (sure foreground)
    _ , sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    # `connectedComponents` will label the background as 0.
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that sure background is 1, not 0
    # The algorithm will use 0 to denote "unlabeled".
    markers = markers + 1
    
    # Mark the region of unknown with 0
    markers[unknown == 255] = 0
    
    # The negative distance transform will be our "elevation map"
    elevation_map = -dist_transform
    
    return markers, elevation_map, dist_transform


# --- Section 3: The Watershed Algorithm Implementation ---
def watershed_from_scratch(elevation_map, markers):
    """
    Implements the watershed algorithm from scratch using a priority queue (min-heap).
    This function "floods" the basins from the initial markers based on the
    elevation map. This version correctly ignores the background label to prevent
    false boundaries.
    
    Args:
        elevation_map (np.array): The landscape to be flooded. Lower values are flooded first.
        markers (np.array): An array with initial seeds. Each seed region has a unique
                            positive integer label > 1. Background is 1, unlabeled is 0.
    
    Returns:
        np.array: The final labeled image, where each basin has a unique integer label
                  and watershed lines are marked with -1.
    """
    h, w = elevation_map.shape
    labels = np.copy(markers)
    
    # Use a min-heap as a priority queue
    pq = []

    # Initialize the queue with pixels at the boundary of the markers
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 1: # Start flooding from object markers only
                is_boundary = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0: continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0:
                            is_boundary = True
                            break
                    if is_boundary: break
                if is_boundary:
                    heapq.heappush(pq, (elevation_map[y, x], x, y))

    # Flooding process
    while pq:
        elevation, x, y = heapq.heappop(pq)
        
        # This pixel was already processed via a shorter path
        if labels[y, x] != 0:
            continue
        
        # Find unique OBJECT labels of processed neighbors.
        # This is the key change: we ignore the background label (1).
        object_neighbor_labels = set()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    # THE FIX IS HERE: Check for labels > 1, not > 0.
                    if labels[ny, nx] > 1:
                        object_neighbor_labels.add(labels[ny, nx])
        
        # If neighbors belong to a single object basin, assign that label
        if len(object_neighbor_labels) == 1:
            label = object_neighbor_labels.pop()
            labels[y, x] = label
            
            # Add its unprocessed neighbors to the queue
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0:
                        heapq.heappush(pq, (elevation_map[ny, nx], nx, ny))

        # If neighbors belong to multiple object basins, this is a true watershed ridge
        elif len(object_neighbor_labels) > 1:
            labels[y, x] = -1

    return labels

# --- Section 4: Visualization and Main Execution Block ---
def visualize_results(original_img, thresh, dist_transform, markers, final_labels):
    """
    Displays the original image and all intermediate and final steps of the
    watershed algorithm using matplotlib.
    """
    # Find the maximum label value to normalize the image for visualization
    max_label = final_labels.max()
    if max_label == 0:
        max_label = 1 # Avoid division by zero if no labels are found

    # Normalize the labels to the 0-255 range to create an 8-bit image
    # We will handle the background (0) and watershed lines (-1) separately after color mapping
    # Note: We perform floating point division before casting to uint8
    normalized_labels = (255 * (final_labels / max_label)).astype(np.uint8)

    # Apply the JET colormap to the normalized labels
    # The input `normalized_labels` is now a valid CV_8UC1 image
    final_labels_display = cv2.applyColorMap(normalized_labels, cv2.COLORMAP_JET)

    # After applying the colormap, explicitly set colors for special regions
    # Make the background (label 0) black
    final_labels_display[final_labels == 0] = [0, 0, 0]
    # Make the watershed lines (label -1) white for high visibility
    final_labels_display[final_labels == -1] = [255, 255, 255]

    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(thresh, cmap='gray')
    axes[1].set_title('Thresholded Image')
    axes[1].axis('off')

    axes[2].imshow(dist_transform, cmap='gray')
    axes[2].set_title('Distance Transform')
    axes[2].axis('off')

    # For displaying markers, use a colormap but ensure background (0) is black
    marker_display = np.copy(markers).astype(np.uint8)
    marker_display[marker_display > 0] = 255 # Make all markers white for clarity
    axes[3].imshow(marker_display, cmap='gray')
    axes[3].set_title('Initial Markers')
    axes[3].axis('off')

    axes[4].imshow(cv2.cvtColor(final_labels_display, cv2.COLOR_BGR2RGB))
    axes[4].set_title('Watershed From Scratch')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 1. Setup a sample image
    IMAGE_PATH = 'hand_drawn_circles.png'
    _ = create_sample_image(IMAGE_PATH) # Create the sample file
    
    # 2. Load and preprocess
    original_image, preprocessed_image = load_and_preprocess_image(IMAGE_PATH)
    
    # 3. Prepare for watershed
    initial_markers, elevation_map, dist_transform_vis = prepare_for_watershed(preprocessed_image)
    
    # 4. Run the from-scratch algorithm
    final_segmentation = watershed_from_scratch(elevation_map, initial_markers)
    
    # 5. Visualize everything
    visualize_results(
        original_img=cv2.imread(IMAGE_PATH),
        thresh=preprocessed_image,
        dist_transform=dist_transform_vis,
        markers=initial_markers,
        final_labels=final_segmentation
    )