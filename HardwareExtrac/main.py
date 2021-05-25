from flask import Flask, render_template, Response, request
import cv2
import time
import os
from lib_camera import GetVideoFrame, SaveVideoFrame
from lib_radar import GetRadarFrame, ConvertRadarFrame, SaveRadarFrame, DisplayRadarFrame
from folder_size import f_size

# Http settings
service_port = 8881
service_ip = '10.42.0.1'
thread = True
stream_url = '/video_feed'
results_url = '/display_face'
cam_url = '/cam_feed'
radar_url = '/radar_feed'
green_url = '/green'
red_url = '/red'
temp_folder = '/home/jetson-nano-tfe/Desktop/tfe_codes/cam_and_radar/.temp_radar'
data_folder = '/home/jetson-nano-tfe/Desktop/tfe_codes/cam_and_radar/data'

current_page = "radar.html"

# Global settings
max_time = 20*60
save_data = True

# Camera settings
fps_camera = 25
cam_id = 0

app = Flask(__name__)
app._static_folder = os.path.dirname(os.path.abspath(__file__)) + "/templates/" # If static images are used flask will search for it in this directory

file_name = os.path.dirname(os.path.abspath(__file__)) + "/templates/not_found.jpg" # Image to display before if current image is None (i.e before first reception)


# Objects creation
camera_frames = GetVideoFrame(cam_id, fps_camera, max_time) 
radar_capture = GetRadarFrame(max_time)
conv_frame = ConvertRadarFrame()
radar_frames = DisplayRadarFrame(conv_frame)

save_video = SaveVideoFrame(camera_frames, data_folder)
save_radar = SaveRadarFrame(radar_capture, data_folder)
     

@app.route('/', methods=['GET', 'POST'])
def index():
    ''' Definie wich page to return for a get request on the root url
    '''
    
    global current_page
    
    if request.method == 'POST': 
    
        if request.form['acquisition'] == 'Start capture':
        
                print('Getting data')
                
                radar_capture.start()
                camera_frames.start()
                conv_frame.start()
                radar_frames.start()
                
                return render_template(current_page)
                
        elif request.form['acquisition'] == 'Stop capture':
        
                print('Stopping capture')
                
                radar_frames.stop()
                camera_frames.stop()
                
                radar_capture.stop()
                conv_frame.stop()
                                
                return render_template(current_page)
                
        elif request.form['acquisition'] == 'Save data':
        
                print('Saving data')
                
                if(f_size(data_folder) > int(50e9)):
                
                        current_page = "memory.html"
                        
                else:
                
                        current_page = "root_green.html"
                        
                        time_0 =  str(time.time())
                        save_video.start(time_0)
                        save_radar.start(time_0)
                
                return render_template(current_page)
                
        elif request.form['acquisition'] == 'Stop saving data':
        
                print('Stop saving data')
                
                current_page = "root_red.html"

                save_video.stop()
                save_radar.stop()
                
                return render_template(current_page)
                
        elif request.form['acquisition'] == 'Connect to radar':
        
                print('Connecting to radar')

                try:
                        radar_capture.connect()
                        time.sleep(5.0)
                        current_page = "root_red.html"
                except:
                        print("Unable to conncet radar")
                        current_page = "radar_error.html"
                                
                return render_template(current_page)
                
        else:
                print('Passed')
                pass
   
    
    if request.method == 'GET': 

        return render_template(current_page) # Root page       

                 
@app.route(cam_url, methods=['GET'])  
def live_cam():

        if request.method == 'GET': 
                return Response(gen_cam(), mimetype='multipart/x-mixed-replace; boundary=frame')
                
@app.route(radar_url, methods=['GET']) # Allow two methods : get and post  
def live_radar():

        if request.method == 'GET': 
                return Response(gen_radar(), mimetype='multipart/x-mixed-replace; boundary=frame')
                        
                       
def gen_cam():

        while True:

                time.sleep(0.07) # Refresh rate

                data = camera_frames.display()

                if(data is None): # If the image is None return the "Not found" image

                        print('[INFO] data camera None')
                        data = cv2.imread(file_name, cv2.IMREAD_COLOR)

                        #data = cv2.imencode('.jpg',picture)[1].tobytes()


                page = (b'--frame\r\n' 
                        b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', data)[1].tobytes() + b'\r\n')

                yield page
                
def gen_radar():

        while True:

                time.sleep(0.05) # Refresh rate

                data = radar_frames.display()

                if(data is None): # If the image is None return the "Not found" image

                        print('[INFO] data radar None')
                        data = cv2.imread(file_name, cv2.IMREAD_COLOR)

                        #data = cv2.imencode('.jpg',picture)[1].tobytes()
                    

                page = (b'--frame\r\n' 
                        b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', data)[1].tobytes() + b'\r\n')

                yield page



if __name__ == '__main__':
    ''' Start the flask application
    '''
    app.run(host=service_ip, port=service_port, threaded=thread)
