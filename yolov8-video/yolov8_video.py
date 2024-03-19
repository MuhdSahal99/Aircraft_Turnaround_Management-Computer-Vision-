from ultralytics import YOLO
from flask import Flask, Response
import cv2
import math
import numpy as np 
import datetime
import requests
import threading


lock = threading.Lock()
api_endpoint ="http://192.168.29.42:8080/tms/events"                                                           #"http://192.168.29.42:8080/tms/events"                                                         #"https://httpbin.org/post"
headers = {
    "Content-Type": "application/json",
}



def make_api_call(event_code, enter_time, exit_time):
    formatted_enter_time = enter_time.strftime('%Y-%m-%dT%H:%M:%S') if enter_time else None
    formatted_exit_time = exit_time.strftime('%Y-%m-%dT%H:%M:%S') if exit_time else None

    payload = {
        "eventInstance": formatted_enter_time,
        "feedDeviceId": 1,
        "eventCode": event_code
    }

    if formatted_exit_time:
        payload["eventInstance"] = formatted_exit_time

    response = requests.post(api_endpoint, json=payload, headers=headers)

    print("API request for", event_code)
    if formatted_enter_time:
        print("Formatted Enter Time:", formatted_enter_time)
    if formatted_exit_time:
        print("Formatted Exit Time:", formatted_exit_time)
    print("Response Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code == 200:
        print("API request successful")
    else:
        print("API request failed")
            
app = Flask(__name__)

@app.route('/')
def index():
    return "Lets Get Started"

def generate_frames():    
    cap=cv2.VideoCapture('C:/Users/HP/Desktop/yolov8/yolov8-video/newzland.mp4')
    frame_width=int(cap.get(3))
    frame_height= int(cap.get(4))

    out=cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

    model=YOLO('C:/Users/HP/Desktop/yolov8/train16/weights/last.pt')

    classNames = ['Aircraft', 'CABIN_AFT_OT', 'CABIN_FWD_OT', 'CAT2_ST', 'CAT_ST', 'CGO_AFT_OT', 'CGO_FWD_OT', 'CHOCKS_ON', 'FUEL_ST', 'PBB_FWD_OFF', 'PBB_FWD_ON', 'PBT_ST']

    aircraft_area = [(660, 126), (275, 207), (535, 629), (1084, 509), (882, 254)] 
    Doors = [(523, 154), (476, 262), (829, 498), (895, 348)]
    Bridge_area = [(882, 270), (894, 585), (1267, 561), (1245, 263), (1039, 200)]
    fueling_area = [(927, 198), (937, 316), (1065, 290), (1045, 172) ]  
    chocks_area = [(863, 494), (877, 583), (1001, 567), (989, 484)]
    pushback_area = [(831, 521), (1027, 703), (1164, 644), (1006, 486)]
    catering_area = [(275, 214), (509, 619), (911, 564), (470, 163)]
    


    enter_time = None
    exit_time = None
    enter_time_bridge_connected = None
    exit_time_bridge_disconnected = None
    frames_inside_aircraft_area = 0
    in_aircraft_area = 0
    max_frames_inside_area = 10
    frames_inside_cargo_door_area = 0 
    frames_inside_bridge_area = 0
    min_frames_to_exit = 30
    enter_time_chocks_on = None
    exit_time_chocks_on = None
    frames_inside_chocks_area = 0

    in_fueling_area = 0 
    max_frames_without_label = 10
    frames_inside_fuel_st_area = 0
    enter_time_fuel_st = None
    exit_time_fuel_st = None
    
    enter_time_cabin_aft_ot = None
    exit_time_cabin_aft_ot = None
    frames_inside_cabin_aft_area = 0

    enter_time_cgo_aft_ot = None
    exit_time_cgo_aft_ot = None
    frames_inside_doors_area = 0

    enter_time_cabin_fwd_ot = None
    exit_time_cabin_fwd_ot = None
    frames_inside_doors_area = 0

    enter_time_cgo_fwd_ot = None
    exit_time_cgo_fwd_ot = None

    enter_time_cat_st = None
    exit_time_cat_st = None
    frames_inside_catering_area = 0

    enter_time_cat2_st = None
    exit_time_cat2_st = None
    frames_inside_catering_area = 0

    enter_time_pbt_st = None
    exit_time_pbt_st = None
    frames_inside_pushback_area = 0

    exit_time = None
# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)                                     

# cv2.namedWindow("Image")
# cv2.setMouseCallback("Image", POINTS)

    while True:
        success, img = cap.read()
        if not success:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    # Doing detections using YOLOv8 frame by frame
    #stream = True will use the generator and it is more efficient than normal
        results=model(img,stream=True)
    #Once we have the results we can check for individual bounding boxes and see how well it performs
    # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
    # we will loop through each of the bouning box
        color_aircraft = (0, 0, 255, 100)  # Red with transparency
        color_fueling = (0, 255, 0, 100)  # Green with transparency
        color_bridge = (255, 0, 0, 100)  # Blue with transparency
        color_doors = (255, 255, 0, 100)
        color_chocks = (255, 0, 255, 100)
        color_pushback = ((0, 165, 255, 100))
        color_catering = ((203, 192, 255, 100))

    
    
    
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
            #print(x1, y1, x2, y2)
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),1) # draw bounding boxes
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                in_aircraft_area = cv2.pointPolygonTest(np.array(aircraft_area, np.int32), (cx, cy), False)
                in_fueling_area = cv2.pointPolygonTest(np.array(fueling_area, np.int32), (cx, cy), False)
                in_bridge_area = cv2.pointPolygonTest(np.array(Bridge_area, np.int32), (cx, cy), False)
                in_doors = cv2.pointPolygonTest(np.array(Doors, np.int32), (cx, cy), False)
                in_chocks_area = cv2.pointPolygonTest(np.array(chocks_area, np.int32), (cx, cy), False)
                in_pushback_area =  cv2.pointPolygonTest(np.array(pushback_area, np.int32), (cx, cy), False)
                in_catering_area = cv2.pointPolygonTest(np.array(catering_area, np.int32), (cx, cy), False)
            
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name = classNames[cls]
                label=f'{class_name}{conf}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            
                if class_name == "Aircraft" and in_aircraft_area >= 0:
                    frames_inside_aircraft_area = 0  # Reset frames counter for valid object
                    if enter_time is None:
                        enter_time =  datetime.datetime.now()
                        formatted_enter_time = enter_time.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"AC_ON_STAND at {formatted_enter_time}")
                        make_api_call("AC_ON_STAND", enter_time, exit_time)
                    
                elif class_name == "Aircraft" and in_aircraft_area < 0:
                    frames_inside_aircraft_area += 1
                    if frames_inside_aircraft_area >= max_frames_inside_area and enter_time is not None:
                        exit_time =  datetime.datetime.now()
                        formatted_exit_time = exit_time.strftime('%Y-%m-%dT%H:%M:%S')
                    # Process the entry and exit timestamps as needed
                        print(f"AC_OFF_STAND at {exit_time}")
                        make_api_call("AC_OFF_STAND", enter_time, exit_time)            
                    
                    
                if class_name == "PBB_FWD_ON" and in_bridge_area >= 0:
                    frames_inside_bridge_area = 0  # Reset frames counter for valid object 
                    if enter_time_bridge_connected is None:
                        current_time = datetime.datetime.now()
                        formatted_enter_time = current_time.strftime('%Y-%m-%dT%H:%M:%S')
                        enter_time_bridge_connected = current_time
                        print(f"Bridge connected  {formatted_enter_time}")
                        make_api_call("PBB_FWD_ON", enter_time_bridge_connected, exit_time_bridge_disconnected)
                    
                if class_name == "PBB_FWD_OFF" and in_bridge_area >= 0:
                    frames_inside_bridge_area += 1
                    if frames_inside_bridge_area >= max_frames_inside_area and enter_time_bridge_connected is not None:
                        current_time = datetime.datetime.now()
                        formatted_exit_time = current_time.strftime('%Y-%m-%dT%H:%M:%S')
                    # Process the entry and exit timestamps as needed
                        print(f"Bridge disconnected at {formatted_exit_time}") 
                        enter_time_bridge_connected = None
                        make_api_call("PBB_FWD_OFF", enter_time_bridge_connected, exit_time_bridge_disconnected)
                    
                if class_name == "CHOCKS_ON":
                    if in_chocks_area >= 0:
                        frames_inside_chocks_area = 0  # Reset frames counter for valid object
                        if enter_time_chocks_on is None:
                            enter_time_chocks_on = datetime.datetime.now()
                            formatted_enter_time = enter_time_chocks_on.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CHOCKS_ON at {formatted_enter_time}")
                            make_api_call("CHOCKS_ON", enter_time_chocks_on, exit_time_chocks_on)
                elif in_chocks_area < 0:
                    frames_inside_chocks_area += 1
                    if frames_inside_chocks_area >= min_frames_to_exit and enter_time_chocks_on is not None:
                        exit_time_chocks_on = datetime.datetime.now()
                        formatted_exit_time = exit_time_chocks_on.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CHOCKS_OFF at {formatted_exit_time}")
                        enter_time_chocks_on = None
                        make_api_call("CHOCKS_OFF", enter_time_chocks_on, exit_time_chocks_on)       
            
        
                if class_name == "FUEL_ST":
                    if in_fueling_area >= 0:
                        frames_inside_fuel_st_area = 0  # Reset frames counter for valid object
                        if enter_time_fuel_st is None:
                            enter_time_fuel_st = datetime.datetime.now()
                            formatted_enter_time = enter_time_fuel_st.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"FUEL_ST at {formatted_enter_time}")
                            make_api_call("FUEL_ST", enter_time_fuel_st, exit_time_fuel_st)   
                                                
                    elif in_fueling_area < 0:
                        frames_inside_fuel_st_area += 1
                        if frames_inside_fuel_st_area >= max_frames_without_label and enter_time_fuel_st is not None:
                            if in_fueling_area < 0:
                                exit_time_fuel_st = datetime.datetime.now()
                                formatted_exit_time = exit_time_fuel_st.strftime('%Y-%m-%dT%H:%M:%S')
                                print(f"FUEL_ET at {formatted_exit_time}")
                                make_api_call("FUEL_ET", enter_time_fuel_st, exit_time_fuel_st)
                                enter_time_fuel_st = None  # Reset enter_time_fuel_st for the next entry
                else:
                    # Reset the counters when the label is not "FUEL_ST"
                    frames_inside_fuel_st_area = 0
                    enter_time_fuel_st = None
                        # exit_time_fuel_st = datetime.datetime.now()
                        # formatted_exit_time = exit_time_fuel_st.strftime('%Y-%m-%dT%H:%M:%S')
                        # print(f"FUEL_ET at {formatted_exit_time}")
                        # enter_time_fuel_st = None
                        # make_api_call("FUEL_ET", enter_time_fuel_st, exit_time_fuel_st)
                        
                if class_name == "CABIN_AFT_OT":
                    if in_doors >= 0:
                        frames_inside_cabin_aft_area = 0  # Reset frames counter for valid object
                        if enter_time_cabin_aft_ot is None:
                            enter_time_cabin_aft_ot = datetime.datetime.now()
                            formatted_enter_time = enter_time_cabin_aft_ot.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CABIN_AFT_OT  at {formatted_enter_time}")
                            make_api_call("CABIN_AFT_OT", enter_time_cabin_aft_ot, exit_time_cabin_aft_ot)

                        
                elif in_doors < 0:
                    frames_inside_cabin_aft_area += 1
                    if frames_inside_cabin_aft_area >= min_frames_to_exit and enter_time_cabin_aft_ot is not None :
                        exit_time_cabin_aft_ot = datetime.datetime.now()
                        formatted_exit_time = exit_time_cabin_aft_ot.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CABIN_AFT_CT at {formatted_exit_time}")
                        enter_time_cabin_aft_ot = None
                        make_api_call("CABIN_AFT_CT", enter_time_cabin_aft_ot, exit_time_cabin_aft_ot)

                         
                if class_name == "CGO_AFT_OT":
                    if in_doors >= 0:
                        frames_inside_doors_area = 0  # Reset frames counter for a valid object
                        if enter_time_cgo_aft_ot is None:
                            enter_time_cgo_aft_ot = datetime.datetime.now()
                            formatted_enter_time = enter_time_cgo_aft_ot.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CGO_AFT_OT  at {formatted_enter_time}")
                            make_api_call("CGO_AFT_OT", enter_time_cgo_aft_ot, exit_time_cgo_aft_ot) 
                elif in_doors < 0:
                    frames_inside_doors_area += 1
                    if frames_inside_doors_area >= min_frames_to_exit and enter_time_cgo_aft_ot is not None:
                        exit_time_cgo_aft_ot = datetime.datetime.now()
                        formatted_exit_time = exit_time_cgo_aft_ot.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CGO_AFT_CT at {formatted_exit_time}")
                        enter_time_cgo_aft_ot = None
                        make_api_call("CGO_AFT_CT", enter_time_cgo_aft_ot, exit_time_cgo_aft_ot)
            
                if class_name == "CABIN_FWD_OT":
                    if in_doors >= 0:
                        frames_inside_doors_area = 0  # Reset frames counter for a valid object
                        if enter_time_cabin_fwd_ot is None:
                            enter_time_cabin_fwd_ot = datetime.datetime.now()
                            formatted_enter_time = enter_time_cabin_fwd_ot.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CABIN_FWD_OT at {formatted_enter_time}")
                            make_api_call("CABIN_FWD_OT", enter_time_cabin_fwd_ot, exit_time_cabin_fwd_ot)
                elif in_doors < 0:
                    frames_inside_doors_area += 1
                    if frames_inside_doors_area >= min_frames_to_exit and enter_time_cabin_fwd_ot is not None:
                        exit_time_cabin_fwd_ot = datetime.datetime.now()
                        formatted_exit_time = exit_time_cabin_fwd_ot.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CABIN_FWD_CT at {formatted_exit_time}")
                        enter_time_cabin_fwd_ot = None
                        make_api_call("CABIN_FWD_CT", enter_time_cabin_fwd_ot, exit_time_cabin_fwd_ot)
                        
                if class_name == "CGO_FWD_OT":
                    if in_doors >= 0:
                        frames_inside_doors_area = 0  # Reset frames counter for a valid object
                        if enter_time_cgo_fwd_ot is None:
                            enter_time_cgo_fwd_ot = datetime.datetime.now()
                            formatted_enter_time = enter_time_cgo_fwd_ot.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CGO_FWD_OT  at {formatted_enter_time}")
                            make_api_call("CGO_FWD_OT", enter_time_cgo_fwd_ot, exit_time_cgo_fwd_ot)
                elif in_doors < 0:
                    frames_inside_doors_area += 1
                    if frames_inside_doors_area >= min_frames_to_exit and enter_time_cgo_fwd_ot is not None:
                        exit_time_cgo_fwd_ot = datetime.datetime.now()
                        formatted_exit_time = exit_time_cgo_fwd_ot.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CGO_FWD_CT  at {formatted_exit_time}")
                        enter_time_cgo_fwd_ot = None
                        make_api_call("CGO_FWD_CT", enter_time_cgo_fwd_ot, exit_time_cgo_fwd_ot)
                        
                          
                if class_name == "CAT_ST":
                    if  in_catering_area  >= 0:
                        frames_inside_catering_area = 0  # Reset frames counter for a valid object
                        if enter_time_cat_st is None:
                            enter_time_cat_st = datetime.datetime.now()
                            formatted_enter_time = enter_time_cat_st.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CAT_ST at {formatted_enter_time}")
                            make_api_call("CAT_ST", enter_time_cat_st, exit_time_cat_st)
                        
                elif  in_catering_area  < 0:
                    frames_inside_catering_area += 1
                    if frames_inside_catering_area >= min_frames_to_exit and enter_time_cat_st is not None:
                        exit_time_cat_st = datetime.datetime.now()
                        formatted_exit_time = exit_time_cat_st.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CAT_ET at {formatted_exit_time}")
                        enter_time_cat_st = None 
                        make_api_call("CAT_ET", enter_time_cat_st, exit_time_cat_st)
            
                if class_name == "CAT2_ST":
                    if  in_catering_area  >= 0:
                        frames_inside_catering_area = 0  # Reset frames counter for a valid object
                        if enter_time_cat2_st is None:
                            enter_time_cat2_st = datetime.datetime.now()
                            formatted_enter_time = enter_time_cat2_st.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"CAT2_ST at {formatted_enter_time}")
                            make_api_call("CAT2_ST", enter_time_cat2_st, exit_time_cat2_st)
                        
                elif  in_catering_area < 0:
                    frames_inside_catering_area += 1
                    if frames_inside_catering_area >= min_frames_to_exit and enter_time_cat2_st is not None:
                        exit_time_cat2_st = datetime.datetime.now()
                        formatted_exit_time = exit_time_cat2_st.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"CAT2_ET at {formatted_exit_time}")
                        enter_time_cat2_st = None
                        make_api_call("CAT2_ET", enter_time_cat2_st, exit_time_cat2_st)  
                        
                if class_name == "PBT_ST":
                    if in_pushback_area >= 0:
                        frames_inside_pushback_area = 0  # Reset frames counter for a valid object
                        if enter_time_pbt_st is None:
                            enter_time_pbt_st = datetime.datetime.now()
                            formatted_enter_time = enter_time_pbt_st.strftime('%Y-%m-%dT%H:%M:%S')
                            print(f"PBT_ST  at {formatted_enter_time}")
                            make_api_call("PBT_ST", enter_time_pbt_st, exit_time_pbt_st)
                        
                elif in_pushback_area < 0:
                    frames_inside_pushback_area += 1
                    if frames_inside_pushback_area >= min_frames_to_exit and enter_time_pbt_st is not None:
                        exit_time_pbt_st = datetime.datetime.now()
                        formatted_exit_time = exit_time_pbt_st.strftime('%Y-%m-%dT%H:%M:%S')
                        print(f"PBT_ET at {formatted_exit_time}")
                        enter_time_pbt_st = None  
                        make_api_call("PBT_ET", enter_time_pbt_st, exit_time_pbt_st)
                        
            ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')            
            
      
        mask = np.zeros_like(img)
    
        cv2.fillPoly(mask, [np.array(aircraft_area, np.int32)], color_aircraft)  # Adjust alpha (fourth value) as needed
        cv2.fillPoly(mask, [np.array(fueling_area, np.int32)], color_fueling)  # Adjust alpha (fourth value) as needed
        cv2.fillPoly(mask, [np.array(Bridge_area, np.int32)], color_bridge)  # Adjust alpha (fourth value) as needed
        cv2.fillPoly(mask, [np.array(Doors, np.int32)], color_doors)
        cv2.fillPoly(mask, [np.array(chocks_area, np.int32)], color_chocks)
        cv2.fillPoly(mask, [np.array(pushback_area, np.int32)], color_pushback)
        cv2.fillPoly(mask, [np.array(catering_area, np.int32)], color_pushback)
    
    
        img = cv2.addWeighted(mask, 0.2, img, 0.8, 0)
    
        # out.write(img)
        # cv2.imshow("Image", img)
    
        if cv2.waitKey(0) & 0xFF==ord('q'):
            break 
    out.release()
    cv2.destroyAllWindows()
    
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

 