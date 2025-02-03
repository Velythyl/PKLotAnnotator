import os
import xml.etree.ElementTree as ET
import cv2
import shutil
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# Load YOLO model once and determine car class ID
model = YOLO("yolo11n.pt").cuda()
car_class_ids = []
for key, value in model.names.items():
    if value in ['car', 'truck', 'bus']:
        car_class_ids += [key]
if len(car_class_ids) == 0:
    raise ValueError("Car class not found in model's class names.")


def call_yolo(image_path, out_main_dir, out_sub_dir):
    """Run YOLO inference and return results plus output directory"""
    results = model(image_path, save=True, project=out_main_dir, name=out_sub_dir)
    return results, os.path.join(out_main_dir, out_sub_dir)


def parse_xml(xml_path):
    """Parse XML and reset all occupancies to 0"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for space in root.findall("space"):
        space.set("occupied", "0")
    return tree, root


def extract_parking_spaces(root):
    """Extract parking spots' contours from XML"""
    spaces = []
    for space in root.findall("space"):
        space_id = space.get("id")
        contour = []

        contourpoints = space.find("contour").findall("point")
        if len(contourpoints) == 0:
            contourpoints = space.find("contour").findall("Point")

        for point in contourpoints:
            x, y = int(point.get("x")), int(point.get("y"))
            contour.append((x, y))

        is_occupied = space.get("occupied")
        if is_occupied is None:
            is_occupied = 0
        is_occupied = int(is_occupied)
        spaces.append((int(space_id), np.array(contour, dtype=np.int32), is_occupied))
    return spaces


def is_overlap(bbox, contour, threshold=0.3):
    """Check if detection overlaps with parking space contour"""
    bbox_poly = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]]
    ], dtype=np.int32)

    intersection = cv2.intersectConvexConvex(bbox_poly, contour)[0]
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    contour_area = cv2.contourArea(contour)

    if bbox_area == 0 or contour_area == 0:
        return False

    overlap_ratio = intersection / min(bbox_area, contour_area)
    return overlap_ratio > threshold


def update_xml_with_detections(xml_path, detections):
    """Update XML with YOLO detections and save as -pred.xml"""
    tree, root = parse_xml(xml_path)
    spaces = extract_parking_spaces(root)

    for bbox in detections:
        for space_id, contour, _ in spaces:
            if is_overlap(bbox, contour):
                for space in root.findall(f"space[@id='{space_id}']"):
                    space.set("occupied", "1")

    new_xml_path = xml_path.replace(".xml", "-pred.xml")
    tree.write(new_xml_path)
    return new_xml_path


def calculate_accuracy(gt_path, pred_path):
    """Calculate occupancy prediction accuracy between two XMLs"""
    gt_tree = ET.parse(gt_path)
    pred_tree = ET.parse(pred_path)

    gt_spaces = {s.get('id'): s.get('occupied') for s in gt_tree.findall('space')}
    pred_spaces = {s.get('id'): s.get('occupied') for s in pred_tree.findall('space')}

    if gt_spaces.keys() != pred_spaces.keys():
        print(f"Space ID mismatch between {gt_path} and {pred_path}")
        return None

    correct = sum(1 for k in gt_spaces if gt_spaces[k] == pred_spaces[k])
    return (correct / len(gt_spaces)) * 100


def move_yolo_image(orig_image_path, yolo_output_dir, dest_folder):
    """Move YOLO output image to destination with -pred suffix"""
    orig_filename = os.path.basename(orig_image_path)
    yolo_image_path = os.path.join(yolo_output_dir, orig_filename)

    if not os.path.exists(yolo_image_path):
        return None

    base, ext = os.path.splitext(orig_filename)
    new_path = os.path.join(dest_folder, f"{base}-pred{ext}")
    shutil.move(yolo_image_path, new_path)
    assert "/" in yolo_output_dir
    shutil.rmtree(yolo_output_dir)
    return new_path


def tree_predict(root_dir):
    """Main processing loop with accuracy tracking"""
    accuracies = []

    for folder, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(folder, file)
            xml_path = os.path.splitext(image_path)[0] + '.xml'

            if not os.path.exists(xml_path):
                print(f"Missing XML for {image_path}")
                print(f"...Deleting all related files for {image_path}")
                os.remove(image_path)
                print("...done.")
                continue

            # YOLO Processing
            out_main_dir = "yolo_results"
            out_sub_dir = os.path.basename(folder)
            results, yolo_dir = call_yolo(image_path, out_main_dir, out_sub_dir)

            # Extract car detections
            detections = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in car_class_ids:
                        detections.append(tuple(map(int, box.xyxy[0].tolist())))

            # XML Processing
            pred_xml_path = update_xml_with_detections(xml_path, detections)

            # Move YOLO output image
            moved_image = move_yolo_image(image_path, yolo_dir, folder)
            if moved_image:
                print(f"Moved prediction image to {moved_image}")
            else:
                print("Failed to move prediction image")
                raise AssertionError()

            # Calculate accuracy
            accuracy = calculate_accuracy(xml_path, pred_xml_path)
            if accuracy is not None:
                accuracies.append(accuracy)
                print(f"Accuracy for {file}: {accuracy:.2f}%")
            else:
                print(f"Accuracy calculation failed for {file}")

    # Final report
    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"\nGlobal Accuracy: {avg:.2f}% ({len(accuracies)} files)")
    else:
        print("No valid accuracy calculations completed")


def cleanup(root_dir):
    """Remove all -pred files in directory tree"""
    count = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if ('-pred.' in file) or (".npz" in file) or (".txt" in file):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                count += 1
                print(f"Removed: {file_path}")
    if os.path.exists("yolo_results"):
        shutil.rmtree("yolo_results")
    print(f"Cleanup complete. Removed {count} files")

def xml_to_array(xml_path, image_filename):
    """Convert XML occupancy data to numpy array with timestamp and sorted spots"""
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract and sort spaces by ID
    spaces = extract_parking_spaces(root)
    spaces.sort(key=lambda x: x[0])
    occupied_values = [occ for (_, _, occ) in spaces]

    # Parse timestamp from image filename
    base_name = os.path.splitext(image_filename)[0]
    parts = base_name.split('_')
    if len(parts) < 4:
        raise ValueError(f"Filename {image_filename} does not match expected format YYYY-MM-DD_HH_MM_SS...")
    date_str = parts[0]
    time_str = f"{parts[1]}:{parts[2]}:{parts[3]}"
    datetime_str = f"{date_str} {time_str}"
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    timestamp = dt.timestamp()

    # Create numpy array: [timestamp, occupied_1, occupied_2, ...]
    array = np.array([timestamp] + occupied_values, dtype=np.float64)
    return array


def tree_xml_to_npz(dataset_root):
    """Generate .npz files with timestamps and occupancy from XML annotations"""
    for root_dir, _, files in os.walk(dataset_root):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            if "-pred." in file:
                continue

            image_path = os.path.join(root_dir, file)
            base_name = os.path.splitext(file)[0]

            # Process Ground Truth XML
            xml_gt_path = os.path.join(root_dir, f"{base_name}.xml")
            if not os.path.exists(xml_gt_path):
                print(f"Missing XML for {image_path}")
                continue

            gt_array = xml_to_array(xml_gt_path, file)
            npz_gt_path = os.path.join(root_dir, f"{base_name}.npz")
            np.savez_compressed(npz_gt_path, occupancy=gt_array)
            print(f"Generated GT NPZ: {npz_gt_path}")

            # Process Prediction XML
            xml_pred_path = os.path.join(root_dir, f"{base_name}-pred.xml")
            if os.path.exists(xml_pred_path):
                pred_array = xml_to_array(xml_pred_path, file)
                assert pred_array[0] == gt_array[0]
                npz_pred_path = os.path.join(root_dir, f"{base_name}-pred.npz")
                np.savez_compressed(npz_pred_path, occupancy=pred_array)
                print(f"Generated Pred NPZ: {npz_pred_path}")


def tree_aggregate(dataset_root):
    """Aggregate per-image NPZ files into per-parking-lot files sorted by time"""
    # Get list of parking lots (top-level directories)
    parking_lots = [d for d in os.listdir(dataset_root)
                    if os.path.isdir(os.path.join(dataset_root, d))]

    for lot in parking_lots:
        lot_dir = os.path.join(dataset_root, lot)

        # Collect all NPZ files in this parking lot's subdirectories
        gt_entries = []
        pred_entries = []

        for root, _, files in os.walk(lot_dir):
            for file in files:
                if root == f"{dataset_root}/{lot}":
                    # ignores previous runs of aggregated data
                    continue

                if file.endswith('.npz'):
                    file_path = os.path.join(root, file)
                    if '-pred.npz' in file:
                        # Prediction file
                        pred_entries.append(file_path)
                    else:
                        # Ground truth file (but ensure it's not a prediction)
                        if not file.endswith('-pred.npz'):
                            gt_entries.append(file_path)

        def process_np_list(nplist):
            np_arrays = []
            for entry in nplist:
                data = np.load(entry)
                np_arrays.append(data['occupancy'])
            # Sort by timestamp (first element)
            gt_arrays_sorted = sorted(np_arrays, key=lambda x: x[0])

            for x in gt_arrays_sorted:
                assert gt_arrays_sorted[0].shape == x.shape

            gt_combined = np.vstack(gt_arrays_sorted)
            return gt_combined

        # Process ground truth
        if gt_entries:
            gt_combined = process_np_list(gt_entries)
            output_path = os.path.join(lot_dir, f"{lot}.npz")
            np.savez_compressed(output_path, occupancy=gt_combined)
            print(f"Aggregated GT for {lot}: {output_path}")
        else:
            print(f"No ground truth NPZ files found for {lot}")

        # Process predictions
        if pred_entries:
            assert gt_entries
            pred_combined = process_np_list(pred_entries)
            assert pred_combined.shape[1] == gt_combined.shape[1]
            output_path = os.path.join(lot_dir, f"{lot}-pred.npz")
            np.savez_compressed(output_path, occupancy=pred_combined)
            print(f"Aggregated Pred for {lot}: {output_path}")
        else:
            print(f"No prediction NPZ files found for {lot}")

        if gt_combined.shape == pred_combined.shape:
            accuracy = (gt_combined == pred_combined).sum() / (gt_combined.shape[0] * gt_combined.shape[1])
            print(f"Accuracy for {lot}: {accuracy}")

            with open(f"{dataset_root}/{lot}/accuracy.txt", "w") as f:
                f.write(f"{accuracy}")
        else:
            with open(f"{dataset_root}/{lot}/accuracy.txt", "w") as f:
                print("Could not compute accuracy because the groundtruth and pred `.npz` were not of the same shape.")
                f.write("Could not compute accuracy because the groundtruth and pred `.npz` were not of the same shape.")


if __name__ == "__main__":
    dataset_root = "/home/charlie/Downloads/parking-lot-dataset/PKLot/PKLot"

    cleanup(dataset_root)

    tree_predict(dataset_root)
    #exit()

    tree_xml_to_npz(dataset_root)

    tree_aggregate(dataset_root)