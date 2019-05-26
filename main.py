import os
from processing import *
from display import draw_circles, show_images
from utils import load_image, resize_segments, save_segments, check_folders, get_average_shape


def create_data(path):
    cropped_array = []
    images = load_image(path, extention='jpg', resize=False)
    for img in tqdm(images):
        pupil_circle = pupil_recognition(img, thresholdpupil=70)
        iris_circle = iris_recognition(img, thresholdiris=160)

        segmented_image, mask = segmentation(
            img, iris_circle, pupil_circle, 90, 180)
        cv2.imshow('Segmented image', segmented_image)

        cropped_image = crop_image(segmented_image, offset=0, tollerance=50)
        cropped_array.append(cropped_image)
        # cv2.imshow('Cropped image', cropped_image)

        draw_circles(img, pupil_circle, iris_circle)
        # show_images(img)

    return cropped_array


def main():
    DATADIR = "./DATA_IMAGES"
    CATEGORIES = ['DB_PROBS', 'DB_NORMAL']
    cropped_dict = {}

    # TODO: Vedere se bisogna cancellare le cartelle di dati ad ogni avvio

    if check_folders(DATADIR) is False:
        raise Exception('Non sono presenti immagini nelle cartelle DB_PROBS e DB_NORMAL')

    for category in tqdm(CATEGORIES):
        data_path = os.path.join(DATADIR, category)
        cropped_dict[category] = create_data(data_path)

    average_shape = get_average_shape(cropped_dict)

    for category in tqdm(CATEGORIES):
        resized_segments = resize_segments(cropped_dict[category], average_shape)
        save_segments(resized_segments, category)


if __name__ == '__main__':
    main()
