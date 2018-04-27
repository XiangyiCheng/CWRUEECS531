Folders:
raw_images: original positive images, Closed_vocal_cord_Expansion, Non_vocal_cord, Open_vocal_cord, collected by Gaojun Jiang.
pos5050: a set of positive images resized and filtered by Closed_vocal_cord_Expansion and Open_vocal_cord.
create_samples5050: postive images created by opencv_createsamples command.
info5050: the images in resized_pos plus images created by opencv_createsamples(including info.lst).
neg: a folder containing all the negative resized and filtered images.
uglies: a folder containing the blank and useless images downloaded from imageNet.
data5050: cascade training result.
test_pos_img: images randomly selected from Closed_vocal_cord_Expansion and Open_vocal_cord to test the training result.
test_neg_img: negative images for testing.
result5050_pos: test result of postive images.
result5050_neg: test result of negative images.

Python Files:
convert_format_n_path: convert the images from original to filtered and resized image. The processed images are stored in different folders.
find_uglies: find blank and useless images from neg folders and delete them.
gather_image: download negative images from imageNet.
resize_pos: resize and filter the postive images which is similar to convert_format_n_path.
create_description_files: create description files for negatives images (compulsive) and postive images (optional) pointing to the images and locating the object.
select_test_img_n_resize: random select and resize the test images for testing the training result from Closed_vocal_cord_Expansion and Open_vocal_cord. 
cv2_test: test code for playing the video in the file.
create_test_image: mix one negative image and one positive image into one image named create_test_img.jpg. Use this image to test the .xml file.
detection: detect the ROI.

Others:
bg.txt: description file created by create_description_files.py for negative images.
positives.vec: vector pointing to positive images. created by opencv_createsamples -info info/info.lst -num 1000 -w 50 -h 50 -vec positives.vec
intubot.JPG: profile shown in GUI.
create_test_img.jpg: created by one negative image and one positive image to test the .xml file.
.xml file: cascade training result. Use this to detect ROI.

Commands:
create more postive samples (including positive description file 'info.lst'):
opencv_createsamples -img pos/655.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle -0.5 -maxzangle 0.5 -num 200

create vector postive images:
opencv_createsamples -info info/info.lst -num 1000 -w 50 -h 50 -vec positives.vec

start haar cascade training:
opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 600 -numNeg 300 -numStages 10 -w 50 -h 50
folder data needs to be created first before runing the command. position needs to point to cascade_training folder.

To test the version of cv2 module:
GO to the opencv virtural environment first, then
(1)python
(2)import cv2
(3)cv2.__version__
After checking the version, exit() to exit python in terminal.

To use cv2, virtual environment have to be used. 
(1)source ~/.profile (2)workon cv