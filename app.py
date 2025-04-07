from ultralytics import YOLO
import json
from collections import defaultdict
import cv2 as cv
import pandas as pd
import gradio as gr
import os
from datetime import datetime

model = YOLO("rest_copilot_6.pt")

names1 = {0: 'American Cheese', 1: 'American Roland Pepper Banana Sliced In Vinegar', 2: 'Black Olives', 3: 'Cappicola', 4: 'Chicken', 5: 'Chocolate Chips Cookies', 6: 'Classic Cheese Dip', 7: 'Concord Grape Jelly', 8: 'Cornbread Cls Mix Stuffing', 9: 'Cranberry', 10: 'Crispy Onion strings', 11: 'Crushed Red Pepper', 12: 'Dip Hummas Traditional', 13: 'Dressing 1000 Island Chef Style', 14: 'Dressing Coleslaw', 15: 'Extra Heavy Mayonnaise', 16: 'French-s Mustard Yellow', 17: 'Ham', 18: 'Horseradish White Prepared', 19: 'Ice Cream', 20: 'Jalapeno Nacho Slices', 21: 'Jellied Cranberry Sauce', 22: 'Lays Sour cream and onion', 23: 'Lemon Juice ready to use', 24: 'Lite Mayonise dressing', 25: 'Mozarella Cheese', 26: 'Oil Olive Blend 8020', 27: 'Onion scliced caramalised frozen', 28: 'Oregano Leaf Whole', 29: 'Original Cheese Dip', 30: 'Parmesan grated cheese', 31: 'Peanut Butter', 32: 'Pepper Black Ground', 33: 'Pepperjack Cheese', 34: 'Pepperoni Sliced Large', 35: 'Pesto Sauce', 36: 'Pickel Dill scliced', 37: 'Provolone Cheese', 38: 'Provolone Cheese sliced', 39: 'Salami', 40: 'Salt', 41: 'Shredded Sauerkraut', 42: 'Sliced Black Olives', 43: 'Spicy Brown Mustard', 44: 'Strawberry Toppings Sliced', 45: 'Sugar Packets', 46: 'Sugar loose', 47: 'Sun Chips Cheddar', 48: 'Sun Chips Garden Salsa', 49: 'Sun Chips Original', 50: 'Turkey', 51: 'Vanilla icecream', 52: 'Water Bottles', 53: 'Whole Bay Leaf', 54: 'chez', 55: 'nac', 56: 'yell'}
names2 = {0: 'American Cheese', 1: 'American Roland Pepper Banana Sliced In Vinegar', 2: 'Beef Corned', 3: 'Beef Roast', 4: 'Black Olives', 5: 'Cappicola', 6: 'Chicken', 7: 'Chocolate Chips Cookies', 8: 'Classic Ground Black Pepper', 9: 'Concord Grape Jelly', 10: 'Cornbread Cls Mix Stuffing', 11: 'Cranberry', 12: 'Crushed Red Pepper', 13: 'Dip Hummas Traditional', 14: 'Dressing 1000 Island Chef Style', 15: 'Dressing Coleslaw', 16: 'Egg Hard Boiled Whole Peeled Select', 17: 'Egg Patty Round', 18: 'Extra Heavy Mayonnaise', 19: 'French mustard yellow', 20: 'French-s Mustard Yellow', 21: 'Fresh iceberg Lettuce', 22: 'Giardiniera Hot Chicago Style', 23: 'Ham', 24: 'Horseradish White Prepared', 25: 'Ice Cream', 26: 'Jalapeno Nacho Slices', 27: 'Jellied Cranberry Sauce', 28: 'Lays Sour cream and onion', 29: 'Lemon Juice ready to use', 30: 'Lettuce', 31: 'Lite Mayonise dressing', 32: 'Mozarella Cheese', 33: 'Oil Olive Blend 8020', 34: 'Oil Olive Blend 90-10', 35: 'Olive Black Ripe Sliced', 36: 'Onion French Crispy Topping', 37: 'Onion Sliced Caramelized Frozen', 38: 'Onion scliced caramalised frozen', 39: 'Oregano Leaf Whole', 40: 'Oreo', 41: 'Original Cheese Dip', 42: 'Parmesan grated cheese', 43: 'Peanut Butter', 44: 'Pepper Banana Sliced', 45: 'Pepper Black Ground', 46: 'Pepper Jalapeno Nacho Sliced Pouch', 47: 'Pepperjack Cheese', 48: 'Pepperoni Sliced Large', 49: 'Pesto Sauce', 50: 'Pickel Dill scliced', 51: 'Pickle Refrigerated Kosher Dill Sliced 25', 52: 'Pickled Onions', 53: 'Potato Fry Crinkle-cut', 54: 'Provolone Cheese', 55: 'Provolone Cheese sliced', 56: 'Salami', 57: 'Salt', 58: 'Sauce Barbecue Original', 59: 'Sauce Pesto Basil', 60: 'Sauce Wing Buffalo', 61: 'Shredded Sauerkraut', 62: 'Sliced Black Olives', 63: 'Spicy Brown Mustard', 64: 'Sriracha', 65: 'Steak', 66: 'Strawberry Toppings Sliced', 67: 'Stuffing Mix Cornbread', 68: 'Sugar Packets', 69: 'Sugar loose', 70: 'Sugar packets', 71: 'Sun Chips Cheddar', 72: 'Sun Chips Garden Salsa', 73: 'Sun Chips Original', 74: 'Tuna Light Premium Pouch No Soy', 75: 'Turkey', 76: 'Vanilla icecream', 77: 'Vinegar Red Wine', 78: 'Water Bottles', 79: 'Whole Bay Leaf', 80: 'cheaz', 81: 'chedder', 82: 'cheesewiz', 83: 'red', 84: 'swiss', 85: 'tri-color cole slaw', 86: 'yell'}
def process_results(main_counter):
    counters = []
    def process_result(json_results):
        class_counts = defaultdict(int)

        for item in json_results:
            if item['confidence'] < 0.5:
                continue
            class_id = item["class"]
            class_counts[class_id] += 1

        counters.append(dict(class_counts))
    for json_results in main_counter:
        process_result(json_results)

    return counters

# Load model
def main_app(vid_path):
    main_counter = []

    cap = cv.VideoCapture(vid_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))  # Get video FPS
    frame_interval = fps  # Process 1 frame per second

    frame_number = 0  # Track frame position

    while cap.isOpened():
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)  # Jump to the correct frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Pass frame directly to your model
        results = model(frame)  # Replace 'model' with your actual model function
        for result in results:
            json_res = result.to_json()
            #convert to list
            json_res = json.loads(json_res)
            main_counter.append(json_res)

        frame_number += frame_interval  # Move to next frame (1 frame per second)

    cap.release()

    counters = process_results(main_counter)

    # Find max count per class
    max_counts = defaultdict(int)
    for counter in counters:
        for class_id, count in counter.items():
            max_counts[class_id] = max(max_counts[class_id], count)

    # Print results
    print("Counters per batch:", counters)
    print("Max count per class:", dict(max_counts))

    # Mapping class numbers to class names
    max_count_names = {names2[class_num]: count for class_num, count in max_counts.items()}

    print(max_count_names)

    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(max_count_names.items()), columns=['Class Name', 'Count'])

    # Add GMT timestamp column
    current_gmt_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df['Timestamp (GMT)'] = current_gmt_time

    # Save as CSV
    csv_path = 'max_class_counts.csv'
    if os.path.exists(csv_path):
        os.remove(csv_path)
    df.to_csv(csv_path, index=False)

    return csv_path
def gradio_interface(video_file):
    if video_file is None:
        return "No file uploaded"
    return main_app(video_file)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.File(label="Download CSV"),
    title="Video Processing App",
    description="Upload a video, process frames at 1 FPS, and get class count CSV."
)


if __name__ == '__main__':
    demo.launch()
