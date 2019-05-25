from processing import *
from display import draw_circles, show_images
from utils import load_image, resize_segment, save_segments


def main(path):
    cropped_array = []
    images = load_image(path, extention='jpg', resize=False)
    for img in tqdm(images):
        pupil_circle = pupil_recognition(img, thresholdpupil=70)
        iris_circle = iris_recognition(img, thresholdiris=160)

        segmented_image, mask = segmentation(
            img, iris_circle, pupil_circle, 30, 50)

        cv2.imshow('Segmented image', segmented_image)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # normalized_img = daugman_normalizaiton(segmented_image, iris_circle, 0,
        #                                        startangle=300, endangle=360)

        cropped_image = crop_image(segmented_image, offset=0, tollerance=50)
        cropped_array.append(cropped_image)

        #cv2.imshow('Cropped image', cropped_image)


        #cv2.imshow('Normalized/Cropped image', normalized_img)

        draw_circles(img, pupil_circle, iris_circle)
        #show_images(img)

    resized_segments = resize_segment(cropped_array)
    save_segments(resized_segments)

if __name__ == '__main__':
    main('./CASIA_DB')
