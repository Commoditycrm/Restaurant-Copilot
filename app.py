from ultralytics import YOLO
import json
from collections import defaultdict
import cv2 as cv
import pandas as pd
import gradio as gr
import os
from datetime import datetime

model = YOLO("rest_copilot_7.pt")
names2 = {0: 'American Cheese', 1: 'American Roland Pepper Banana Sliced In Vinegar', 2: 'Beef Corned', 3: 'Beef Roast', 4: 'Black Olives', 5: 'Cappicola', 6: 'Chicken', 7: 'Chocolate Cookies', 8: 'Classic Ground Black Pepper', 9: 'Concord Grape Jelly', 10: 'Cornbread Cls Mix Stuffing', 11: 'Cranberry', 12: 'Crushed Red Pepper', 13: 'Dip Hummas Traditional', 14: 'Dressing 1000 Island Chef Style', 15: 'Dressing Coleslaw', 16: 'Egg Hard Boiled Whole Peeled Select', 17: 'Frenchs Mustard Yellow', 18: 'Fresh iceberg Lettuce', 19: 'Giardiniera Hot Chicago Style', 20: 'Ham', 21: 'Horseradish White Prepared', 22: 'Ice Cream', 23: 'Jalapeno Nacho Slices', 24: 'Lays Sour cream and onion', 25: 'Lemon Juice ready to use', 26: 'Lettuce', 27: 'Lite Mayonise dressing', 28: 'Meat Gyro Slices Pre-cooked', 29: 'Mozarella Cheese', 30: 'Mustard Dijon', 31: 'Mustard HoneyDijon', 32: 'Oil Olive Blend 8020', 33: 'Oil Olive Blend 90-10', 34: 'Olive Black Ripe Sliced', 35: 'Onion French Crispy Topping', 36: 'Onion Sliced Caramelized Frozen', 37: 'Onion scliced caramalised frozen', 38: 'Oregano Leaf Whole', 39: 'Oreo', 40: 'Original Cheese Dip', 41: 'Parmesan grated cheese', 42: 'Peanut Butter', 43: 'Pepper Banana Sliced', 44: 'Pepper Jalapeno Nacho Sliced Pouch', 45: 'Pepperjack Cheese', 46: 'Pepperoni Sliced Large', 47: 'Pesto Sauce', 48: 'Pickel Dill scliced', 49: 'Pickle Refrigerated Kosher Dill Sliced 25', 50: 'Pickled Onions', 51: 'Potato Fry Crinkle-cut', 52: 'Provolone Cheese', 53: 'Provolone Cheese sliced', 54: 'Salami', 55: 'Salt', 56: 'Sauce Barbecue Original', 57: 'Sauce Marinara Premium Midwest', 58: 'Sauce Wing Buffalo', 59: 'Shredded Sauerkraut', 60: 'Sliced Black Olives', 61: 'Spicy Brown Mustard', 62: 'Sriracha', 63: 'Steak', 64: 'Strawberry Toppings Sliced', 65: 'Stuffing Mix Cornbread', 66: 'Sugar loose', 67: 'Sugar packets', 68: 'Sun Chips Cheddar', 69: 'Sun Chips Garden Salsa', 70: 'Sun Chips Original', 71: 'Tuna Light Premium Pouch No Soy', 72: 'Turkey', 73: 'Vanilla icecream', 74: 'Vinegar Red Wine', 75: 'Water Bottles', 76: 'Whole Bay Leaf', 77: 'cappicola_r', 78: 'chedder', 79: 'cheesewiz', 80: 'swiss', 81: 'tri-color cole slaw'}

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
