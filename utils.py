# import the necessary packages
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os

def load_perf_data(directory, df, im_size):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    the_class = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            #print(df["ID"].values)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                if int(folder_strip) in df["ID"].values:
                    dir_path = os.path.join(directory, folder)
                    # Loading images
                    file_name = os.path.basename(file)[0]
                    if file_name == 'b':
                        img1 = mpimg.imread(os.path.join(dir_path, file))
                        img1 = resize(img1, (im_size, im_size))
                    elif file_name == 'm':
                        img2 = mpimg.imread(os.path.join(dir_path, file))
                        img2 = resize(img2, (im_size, im_size))
                    elif file_name == 'a':
                        img3 = mpimg.imread(os.path.join(dir_path, file))
                        img3 = resize(img3, (im_size, im_size))

                        out = cv2.vconcat([img1, img2, img3])
                        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                        gray = resize(gray, (224, 224))
                        out = cv2.merge([gray, gray, gray])
                        #out = gray[..., np.newaxis]
                        #out = np.array(out)
                        images.append(out)

                        # Defining labels
                        if df[df["ID"].values == int(folder_strip)]['MVD'].values == 1:
                            the_class.append(1)
                        elif df[df["ID"].values == int(folder_strip)]['epi_cad'].values == 1:
                            the_class.append(2)
                        else:
                            the_class.append(3)

    return (np.array(images), np.array(the_class))

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
	    # store the target image width, height, and interpolation
	    # method used when resizing
	    self.width = width
	    self.height = height
	    self.inter = inter

    def preprocess(self, image):
	    # resize the image to a fixed size, ignoring the aspect
	    # ratio
	    return cv2.resize(image, (self.width, self.height),interpolation=self.inter)


class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)


class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))
