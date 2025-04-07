from ultralytics import YOLO

model = YOLO("rest_copilot_5.pt")

results = model("Capture2.PNG")

for result in results:
    print(result)
    result.save()