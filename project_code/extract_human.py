import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util


cap = cv2.VideoCapture(r'E:\emotion_recognition\videos\videos_sum\201811_V1.mp4')
ret, image_np = cap.read()
out = cv2.VideoWriter(r'E:\emotion_recognition\videos\videos_sum\output_video.mp4', -1, cap.get(cv2.CAP_PROP_FPS), (image_np.shape[1], image_np.shape[0]))

PATH_TO_CKPT = './ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
save_path='./img/'
i = 0

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        while cap.isOpened():
            ret, image_np = cap.read()
            if len((np.array(image_np)).shape) == 0:
                break

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # print(boxes[0].shape)
            # print(boxes[0][1])
            print(image_np.shape)
            print(scores[0][0])
            print(classes[0][0])
            box = boxes[0][0]
            ymin = int(box[0] * image_np.shape[0])
            xmin = int(box[1] * image_np.shape[1])
            ymax = int(box[2] * image_np.shape[0])
            xmax = int(box[3] * image_np.shape[1])
            # print("picture:",image_np.shape)
            # print("xmin",xmin)
            # print("xmax:", xmax)
            # print("ymin:",ymin)
            # print("ymax:",ymax)
            if scores[0][0]>=0.6 and classes[0][0]==1:
                new_img = cv2.rectangle(image_np, (xmin,ymin),(xmax,ymax), (0, 0, 255), 2)
                img = image_np[ymin:ymax,xmin:xmax]
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                cv2.imwrite('E:/emotion_recognition/pictures/img/photo_{}.jpg'.format(i), img)
                i+=1
            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)
            out.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

cap.release()
out.release()
cv2.destroyAllWindows()
