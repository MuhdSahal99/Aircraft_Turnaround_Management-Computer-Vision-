from ultralytics import YOLO
import cv2
import math
import numpy as np
import datetime

cap = cv2.VideoCapture('C:/Users/HP/Desktop/yolov8/yolov8-video/newzland.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO('C:/Users/HP/Desktop/yolov8/train7/weights/last.pt')

classNames = ["Aft starboard door", "Aircraft", "Bridge disconnected", "Bridge", "Bridge_connected", "Cargo_door_opened", "FWD starboard door", "Fueling started", "Pushback truck", "cargo_door_closed", "fueling stopped", "fwd starboard door closed", "fwd starboard door opened", "person"]
door_labels = ["Aft starboard door", "FWD starboard door", "cargo_door_closed", "fwd starboard door closed", "fwd starboard door opened", "Cargo_door_opened"]
fueling_labels = ["Fueling started", "fueling stopped"]
bridge_labels = ["Bridge_connected","Bridge disconnected"]

aircraft_stationary = [(307, 198), (585, 586), (1128, 475), (847, 237), (665, 125)]
Doors = [(529, 191), (509, 259), (855, 490), (888, 352)]
Bridge_area = [(890, 296), (1246, 266), (1269, 564), (950, 589), (891, 297)]
fueling_area = [(943, 299), (922, 202), (1013, 183), (1064, 273)]

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return hours, minutes, seconds

MIN_EXIT_DURATION = 10.0  # Minimum duration for an exit to be considered valid
object_trackers = {}

enter_times = {}
bridge_areas = {label_key: {} for label_key in bridge_labels}
object_positions = {}
exit_frame_counters = {}

door_enter_times = {}
fueling_enter_times = {}
exit_times = {}

while True:
    success, img = cap.read()
    if not success:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            in_aircraft = cv2.pointPolygonTest(np.array(aircraft_stationary, np.int32), (cx, cy), False)
            in_fueling = cv2.pointPolygonTest(np.array(fueling_area, np.int32), (cx, cy), False)
            in_bridge = cv2.pointPolygonTest(np.array(Bridge_area, np.int32), (cx, cy), False)
            in_doors = cv2.pointPolygonTest(np.array(Doors, np.int32), (cx, cy), False)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'

            if in_aircraft >= 0:
                label_key = "Aircraft"
                if label_key not in enter_times:
                    enter_times[label_key] = current_time
                    print(f'{label_key} entered the aircraft area at {current_time:.2f} seconds')
                elif class_name == label_key and label_key in enter_times:
                    enter_time = enter_times[label_key]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration = exit_time - enter_time
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        print(f'{label_key} exited the aircraft area at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}')
                        exit_times[label_key] = current_time
                        del enter_times[label_key]
            else:
                label_key = "Aircraft"
                if label_key in enter_times:
                    enter_time = enter_times[label_key]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration = exit_time - enter_time
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        print(f'{label_key} exited the aircraft area at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}')
                        exit_times[label_key] = current_time
                        del enter_times[label_key]

            # ... (similar logic for other areas)

            # Object tracking
            if label not in object_trackers:
                object_trackers[label] = cv2.TrackerCSRT_create()
                bbox = (x1, y1, x2 - x1, y2 - y1)
                object_trackers[label].init(img, bbox)
            else:
                success, new_bbox = object_trackers[label].update(img)
                if success:
                    (x, y, w, h) = tuple(map(int, new_bbox))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # Tracking failure, remove tracker
                    del object_trackers[label]
                
            if in_fueling >= 0:
                label_key = "Fueling started"
                if label_key not in enter_times:
                    enter_times[label_key] = current_time
                    print(f'{label_key} entered the fueling_area at {current_time:.2f} seconds')
                elif class_name == label_key and label_key in enter_times:
                    enter_time = enter_times[label_key]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration = exit_time - enter_time
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        print(f'{label_key} exited the fueling_area at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}')
                        exit_times[label_key] = current_time
                        del enter_times[label_key]
            else:
                label_key = "Fueling started"
                if label_key in enter_times:
                    enter_time = enter_times[label_key]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration = exit_time - enter_time
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        print(f'{label_key} exited the fueling_area at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}')
                        exit_times[label_key] = current_time
                        del enter_times[label_key]
                        
            for label_key in bridge_areas:
                if label in bridge_areas[label_key]:
                    enter_time = bridge_areas[label_key][label]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration_seconds = exit_time - enter_time
                        hours, minutes, seconds = seconds_to_hms(duration_seconds)
                        current_system_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f'{label} exited the {label_key} area at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}. Time: {current_system_time}')
                        del bridge_areas[label_key][label]
                elif in_bridge >= 0 and class_name == label_key and label not in bridge_areas[label_key]:
                    bridge_areas[label_key][label] = current_time
                    current_system_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f'{label} entered the {label_key} area at {current_time:.2f} seconds. System Time: {current_system_time}')
 


            if in_doors >= 0:
                label_key = "Cargo_door_opened"
                if label_key not in enter_times:
                    enter_times[label_key] = current_time
                    print(f'{label_key} entered the Doors at {current_time:.2f} seconds')
                elif class_name == label_key and label_key in enter_times:
                    enter_time = enter_times[label_key]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration = exit_time - enter_time
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        print(f'{label_key} exited the Doors at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}')
                        exit_times[label_key] = current_time
                        del enter_times[label_key]
            else:
                label_key = "Cargo_door_opened"
                if label_key in enter_times:
                    enter_time = enter_times[label_key]
                    if current_time - enter_time >= MIN_EXIT_DURATION:
                        exit_time = current_time
                        duration = exit_time - enter_time
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        print(f'{label_key} exited the Doors at {current_time:.2f} seconds. Duration: {hours}:{minutes}:{seconds}')
                        exit_times[label_key] = current_time
                        del enter_times[label_key]
                        
            

            # Similar logic for other areas (fueling, bridge, doors)

    mask = np.zeros_like(img)

    cv2.fillPoly(mask, [np.array(aircraft_stationary, np.int32)], (0, 0, 255, 100))
    cv2.fillPoly(mask, [np.array(fueling_area, np.int32)], (0, 255, 0, 100))
    cv2.fillPoly(mask, [np.array(Bridge_area, np.int32)], (255, 0, 0, 100))
    cv2.fillPoly(mask, [np.array(Doors, np.int32)], (255, 255, 0, 100))

    img = cv2.addWeighted(mask, 0.2, img, 0.8, 0)

    out.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
