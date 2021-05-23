import pickle 
from multiprocessing import Process, Queue
import time 
import numpy as np
import cv2
import datetime
import threading 
import socket
import struct
import array
from radc_frame import radar_frame
from matplotlib import pyplot as plt
from threading import Thread
import pickle
import os, fnmatch
from PIL import Image
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

class GetRadarFrame:

        def __init__(self, t_max, temp_folder = '/home/jetson-nano-tfe/Desktop/tfe_codes/cam_and_radar/.temp_radar'):
        
                self.temp_folder = temp_folder
                self.t_max = t_max
                self.stopped = False
                self.Q = Queue(maxsize=5)


             
        def connect(self):
               
                #BW = 970 # 40 m max, 0.157 resol
                BW = 554 # 70 m max, 0.2749 resol
                
                f_0 = 23848 #40m, 23848 70m, 23800 40m
                f_down = int(f_0 - BW / 2)
                
                delay = 2214 # 80 km/h max, 0.63 resol
                #delay = 1147 # 100 km/h max, 0.787 resol

                print('[INFO] Radar parameters setting')

                TCP_IP = '192.168.16.2'
                TCP_PORT = 6172

                try:   
                        self.radar_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.radar_socket.settimeout(30.0)
                        self.radar_socket.connect((TCP_IP, TCP_PORT))
                        print ('[INFO] connected to radar')
                except socket.error as error:
                        print('[ERROR] Unable to connect to radar : ')
                        print(error)
                        raise Exception('Connection to radar failed')

                initial_wait_time = 0.2
                wait_time = 0.05 # Pas certain que ca change quelque chose au final...
                final_wait_time = 0.5

                time.sleep(initial_wait_time)

                #MESSAGE = b'INIT' + struct.pack("<I", 0)
                #s.send(MESSAGE)
                #time.sleep(wait_time)
                try:
                        MESSAGE = b'INIT' + struct.pack("<I", 0)
                        #self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        psnp = struct.pack("<I", 200)
                        MESSAGE = b'PSNP' + struct.pack("<I", 4) + psnp
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        rangecomp = struct.pack("<f", 0.0)
                        MESSAGE = b'PSRC' + struct.pack("<I", 4) + rangecomp
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        psbr = struct.pack("<I", 2)
                        MESSAGE = b'PSBR' + struct.pack("<I", 4) + psbr
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        pstr = struct.pack("<I", 200)
                        MESSAGE = b'PSTR' + struct.pack("<I", 4) + pstr
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        psbs = struct.pack("<I", 0)
                        MESSAGE = b'PSBS' + struct.pack("<I", 4) + psbs
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        psts = struct.pack("<I", 100)
                        MESSAGE = b'PSTS' + struct.pack("<I", 4) + psts
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        #pspt = struct.pack("<f", 0.0)
                        pspt = struct.pack("<f", 1000)
                        MESSAGE = b'PSPT' + struct.pack("<I", 4) + pspt
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        rsrg = struct.pack("<I", 20)
                        MESSAGE = b'RSRG' + struct.pack("<I", 4) + rsrg
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        psbu = struct.pack("<I", 128)
                        MESSAGE = b'PSBU' + struct.pack("<I", 4) + psbu
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        rssf = struct.pack("<I", f_0)
                        MESSAGE = b'RSSF' + struct.pack("<I", 4) + rssf
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        rsbw = struct.pack("<I", BW)
                        MESSAGE = b'RSBW' + struct.pack("<I", 4) + rsbw
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        rsid = struct.pack("<I", delay)
                        MESSAGE = b'RSID' + struct.pack("<I", 4) + rsid
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)
                        
                        psso = struct.pack("<I", 0)
                        MESSAGE = b'PSSO' + struct.pack("<I", 4) + psso
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'RPRM'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'PPRM'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'RADC'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'RDBS'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'RDDA'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'RMRD'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'PLEN'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'PDAT'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)


                        MESSAGE = b'DSF0' + struct.pack("<I", 4) + b'TLEN'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(wait_time)

                        MESSAGE = b'DSF1' + struct.pack("<I", 4) + b'RADC'
                        self.radar_socket.send(MESSAGE)
                        time.sleep(final_wait_time)

                except:

                        self.quit_radar()
                   
        def stop(self):
        
                self.stopped = True
                
        def quit_radar(self):
        
            self.stopped = True
        
            try:

                discon_wait_time = 1.0

                MESSAGE = b'GBYE' + struct.pack("<I", 0)
                self.radar_socket.send(MESSAGE)
                time.sleep(discon_wait_time)

                print('[INFO] Radar server disconnected')

                self.radar_socket.close()

                print('[INFO] Socket closed')

            except:
                print('[ERROR] Unable to disconnecte radar server')
        
                
        def start(self):
                self.stopped = False
                t = Thread(target=self.update, args=())
                t.start() 
                return self
                
        def update(self):
        
                t_in = time.time()
                
                max_size = 8*4096
                
                data = bytearray(max_size)
                buff = memoryview(bytearray(max_size))

                length = len(data)
  
                while(data.find(b'RADC') < 0): # Commencer avec un RADC
                        num = self.radar_socket.recv_into(buff)
                        data += buff[0:length]

                data = data[data.find(b'RADC'):len(data)] # Erase data before first PDAT
                
                t_f = time.time()
                
                t_find = 0.0
                
                first = True
                                
                while(self.t_max>time.time()-t_in):
                
                        if(self.stopped):
                                return
                        
                        num = self.radar_socket.recv_into(buff)
                        data += buff[0:length]
                        
                        header_position = data.find(b'RADC')
                        tail_position = data.find(b'DONE')
                        
                        if(header_position >= 0 and first):
                                first = False
                                t_acc = time.time()

                        
                        if(header_position >= 0 and tail_position >= 0):
                                
                                fr = data[header_position + 8:tail_position]
                                
                                if not self.Q.full():
                                        self.Q.put((fr, t_acc))
                                else:
                                        print('[Warning] Queue for radar frames is full')
                                        
                                first = True
                                
                                filename_temp = str(self.temp_folder) + '/frame_' + str(time.time()) + '.temp'
                                filename = str(self.temp_folder) + '/frame_' + str(time.time()) + '.pickle'
                                pickle.dump((fr, time.time()), open(filename_temp, 'wb'))
                                os.rename(filename_temp, filename)
                                
                                t_nf = time.time() - t_f
                                if(t_nf > 0.11):
                                        print('New frame : ', t_nf)
                                        
                                t_f = time.time()
                                        
                                data = data[tail_position + 8:]
                                
        def read(self):
                return self.Q.get()
                
                
                        
class ConvertRadarFrame:

        def __init__(self, temp_folder = '/home/jetson-nano-tfe/Desktop/tfe_codes/cam_and_radar/.temp_radar'): 
       
                self.stopped = False 
                self.pattern = "*.pickle"
                self.temp_folder = temp_folder
                self.q = Queue(maxsize=3)
                
        def start(self):
                self.stopped = False
                t = Thread(target=self.update, args=())
                t.start() 
                return self
                
        def stop(self):
        
                self.stopped = True
                
                
        def update(self):
        
                i = 0
                
                while True:
                                
                        listOfFiles = os.listdir(str(self.temp_folder) + '/.')  
                        
                        if(self.stopped and not listOfFiles):
                                return
                                
                        j = 0 
                        
                        for frame in listOfFiles:
                                if fnmatch.fnmatch(frame, self.pattern):
                                
                                        j+=1
                                        
                                        if(j>2):
                                                print("Conversion delayed : ", j)
                                
                                        t_00 = time.time()
                                        
                                        filename = str(self.temp_folder) + '/' + frame
                                
                                        d = pickle.load(open(filename, 'rb'))
                                        os.remove(filename)
                        
                                        i += 1 
                                        
                                        data = d[0]
                                        
                                        z = array.array("H", data)
                                        z = np.array(z, dtype='complex')

                                        data = z[0::2] + 1j * z[1::2]

                                        data_cal = data[2:]

                                        try:
                                                
                                                Z1C = data_cal[0:256 * 256].reshape((256, 256))
                                                #Z2C = data_cal[256 * 256:2 * 256 * 256].reshape((256, 256))
                                                #Z3C = data_cal[256 * 256 * 2:3 * 256 * 256].reshape((256, 256))
                                                
                                                #Z1C = (Z1C + Z2C + Z3C)/3
                                                
                                                

                                                cal_ZZ_1 = np.mean(Z1C, axis=0)
                                                #cal_ZC_1 = np.mean(Z1C, axis=1)
                                                #cal_ZZ_2 = np.mean(Z2C, axis=0)
                                                #cal_ZZ_3 = np.mean(Z3C, axis=0)
                                                
                                                range_speed_ant1 = np.fft.fft2(Z1C - cal_ZZ_1) #- cal_ZC_1)
         
                                                fshift = np.fft.fftshift(range_speed_ant1, 0)
                                                
                                                try:
                                                        magnitude_spectrum = 20*np.log(np.abs(fshift))
                                                except: 
                                                        print('[INFO] Ok')
                                                        
                                                if not self.q.full():
                                                        self.q.put(magnitude_spectrum)
                                                        
                                                else:
                                                        self.q.get()
                                                        self.q.put(magnitude_spectrum)
                                                        print("Frame not displayed")
                                                        
                                                if(time.time() - t_00 > 0.2):
                                                        
                                                        print("Total time : ", time.time() - t_00)
                                                        
                                        except:
                                                print('Frame not converted')
                                                
               
        def display(self):
                return self.q.get()

class SaveRadarFrame:

        def __init__(self, getFrame, folder):
                self.stopped = False
                self.getFrame = getFrame
                self.folder = folder

        def start(self, filename):
                self.stopped = False
                t = Thread(target=self.save, args=((filename,)))
                #t.daemon = True
                t.start() 
                return self
                
        def save(self, filename):
        
                f = self.folder + '/' + filename + '.pickle'
                fich = open(f, 'wb')
                
                while (not self.getFrame.stopped or not self.getFrame.Q.empty()):
                
                        if(self.stopped):
                                fich.close()
                                return
                                
                        if not self.getFrame.Q.empty():           
                                frame = self.getFrame.read()  
                                pickle.dump(frame, fich)         
                                
                        else:
                                time.sleep(0.02)
                        
                fich.close()
                
        def stop(self):
                self.stopped = True
                
                        
class DisplayRadarFrame:

        def __init__(self, convFrame):
                self.stopped = False
                self.convFrame = convFrame
                self.current = None

        def start(self):
                self.stopped = False
                t = Thread(target=self.convert, args=())
                t.start() 
                return self
                
        def stop(self):
        
                self.stopped = True
                
        def convert(self):
              
                while (not self.convFrame.stopped or not self.convFrame.q.empty()):
                        
                        if(self.stopped):
                                return  
                
                        if not self.convFrame.q.empty():    
                        
                        
                                frame = self.convFrame.display()
                        
                                m = np.amax(frame)
                                factor = m/255.0
                                 
                             
                               	frame[:] = [x / factor for x in frame]
                                #number_of_white = 0
                                
                                #for i in range(0,255):
                                #        for j in range(0,255):     
                                #                if(frame[i,j] < 195.0):
                                #                        number_of_white += 1

                                if False: #number_of_white > 256*256*0.4:
                                        print("Image not displayed")
                           
                                else: 
                                	neighborhood = generate_binary_structure(2,2)
                                
                                	frame = maximum_filter(frame, footprint=neighborhood, mode='mirror') 
                                
                                	#local_max = maximum_filter(frame, footprint=neighborhood) == frame
                                
                                	#background = (frame == 0.0)
                                
                                	#eroded_background = binary_erosion(background, structure = neighborhood, border_value=1)
                                
                                	#frame = local_max ^ eroded_background
                                
                                	new_im = np.ndarray((3, 256, 256), dtype = int)
                                	new_im[0::] = frame
                                	new_im = new_im.astype(np.uint8)
                                	new_im_red, new_im_green, new_im_blue = new_im
                               
                                	new_im_gray = np.dstack([new_im_red, new_im_green, new_im_blue])
                                	cv2.cvtColor(new_im_gray, cv2.COLOR_BGR2GRAY)
                                
                                	ret, im_th = cv2.threshold(new_im_gray, 240, 255, cv2.THRESH_BINARY)
                                	#ret, im_th = cv2.threshold(new_im_gray, 195, 0, cv2.THRESH_TOZERO)

                                	#im_th = new_im_gray
                                
                                
                                	im_th = cv2.resize(im_th, (0,0), fx = 2.0, fy = 2.0, interpolation = cv2.INTER_LINEAR)
                              
                                	self.current = im_th
                               
                        else:
                                time.sleep(0.02)

        def display(self):
        
                return self.current

