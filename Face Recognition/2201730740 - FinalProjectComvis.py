# Tonto Claudinus
# 2201730740
# BA01
# Final Project - COmputer Vision

def get_path_list(root_path):
    import os

    actor_name = os.listdir(root_path)
    return actor_name
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

def get_class_names(root_path, train_names):
    import os

    list_all_image = []
    list_image_class = []
    

    for label, actor_names in enumerate(train_names):
        full_name_path = root_path + '/' + actor_names

        for actor_path in os.listdir(full_name_path):
            all_image_path = full_name_path + '/' + actor_path
            list_all_image.append(all_image_path)
            list_image_class.append(label)
    
    return list_all_image, list_image_class
    
    '''
        To get a list of train image and a list of image class

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image class
    '''

def detect_faces_and_filter(image_list, image_classes_list=None):
    import cv2
    import os
    import numpy as np

    faces_list  = []
    faces_label_list = []
    faces_loc = []

    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
     
    if (image_classes_list == None):
        image_classes_list = np.arange(0, len(image_list)) 

    for label, image in zip(image_classes_list, image_list):

        gray_image = cv2.imread(image, 0)

        detected_faces = cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)

        if(len(detected_faces) < 1):
            continue
        for detected_face in detected_faces:
            x, y, w, h = detected_face
            face = gray_image[y:y+w, x:x+h]

            faces_list.append(face)
            faces_loc.append(detected_face)
            faces_label_list.append(label)

    return faces_list, faces_loc, faces_label_list

    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image class
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image class
    '''

def train(train_face_grays, image_classes_list):
    import cv2
    import numpy as np

    face_recog = cv2.face.LBPHFaceRecognizer_create()
    face_recog.train(train_face_grays, np.array(image_classes_list))

    return face_recog
    
    '''
        To create and train recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path):
    import os

    list_all_test_images = []

    for image_path in os.listdir(test_root_path):
        test_image_path = test_root_path + '/' + image_path
        list_all_test_images.append(test_image_path)

    return list_all_test_images
    
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded test images
    '''


def predict(recognizer, test_faces_gray):
    
    pred_res = []

    for test_image in test_faces_gray:
        label,_ = recognizer.predict(test_image)
        pred_res.append(label)
    
    return pred_res
    
    '''
        To predict the test image with recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    import cv2

    predicted_test_image_list = []

    names = []

    for name in train_names:
        names.append(name)

    for test_images, loc, result in zip(test_image_list, test_faces_rects, predict_results):

        images = cv2.imread(test_images)

        x, y, w, h = loc

        cv2.rectangle(images, (x,y), (x+w, y+h), (0, 0, 255), 8)
        
        text = names[result]
        cv2.putText(images, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        predicted_test_image_list.append(images)

    return predicted_test_image_list

    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''
    
def combine_and_show_result(image_list):
    import cv2

    final_images_resize = [cv2.resize(images, (200, 200), interpolation=cv2.INTER_LINEAR) for images in image_list]
    final_image = cv2.hconcat(final_images_resize)
    cv2.imshow('Final Result', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

    '''
        To show the final image that already combine into one image
        Before the image combined, it must be resize with
        width and height : 200px

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    print(filtered_classes_list)
    # recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    # test_names = get_path_list(test_root_path)
    # test_image_list = get_test_images_data(test_root_path)
    # test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    # predict_results = predict(recognizer, test_faces_gray)
    # predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    # combine_and_show_result(predicted_test_image_list)