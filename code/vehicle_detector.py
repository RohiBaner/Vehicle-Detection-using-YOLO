from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt

execution_path = os.getcwd()

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}

resized = False

input_path = "../data/test1.mp4"
output_path = "../data/test1_output1"

def forSecond(frame_number, output_arrays, count_arrays, average_count, returned_frame):

    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0

    for eachItem in average_count:
        counter += 1
        labels.append(eachItem + " = " + str(average_count[eachItem]))
        sizes.append(average_count[eachItem])
        this_colors.append(color_index[eachItem])

    global resized

    if (resized == False):
        manager = plt.get_current_fig_manager()
        manager.resize(width=1200, height=600)
        resized = True

    plt.subplot(1, 2, 1)
    plt.title("Second : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)

##Variable to contain all the bounding boxes for each object detected
#box_points = list()
#def forFrame(frame_number, output_array, output_count):
#    global box_points
#    #Get the bounding box
#    for i in range(len(output_array)):
#        #Store bounding boxes of only persons
#        if output_array[i]['name'] == 'person':
#            box_points.append(output_array[i]['box_points'])


#execution_path = os.getcwd()
detector = VideoObjectDetection()
#detector.setModelTypeAsRetinaNet()
#detector.setModelTypeAsTinyYOLOv3()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "../models/yolo.h5"))
detector.loadModel(detection_speed="fast")

#We can use the forframe function to extract objects from video
'''video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, input_path),
                                output_file_path=os.path.join(execution_path, output_path)
                                , frames_per_second=60, per_frame_function=forFrame, log_progress=True)'''
plt.show()
#Simple object detection in video
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, input_path),
                                output_file_path=os.path.join(execution_path, output_path)
                                , frames_per_second=30, log_progress=True)