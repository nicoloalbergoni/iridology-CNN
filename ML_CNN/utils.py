import os


def check_folders(datadir):
    """
    Checks if all the necessary folders exists otherwise it creates them.
    It also checks if there are images in those folders.

    :param datadir: path of the main directory
    :type datadir: str
    :return: True if there are images in the folders otherwise returns false
    :rtype: bool
    """
    file_count_normal = 0
    file_count_probs = 0
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    probs_path = os.path.join(datadir, 'DB_PROBS_SEG')
    if not os.path.exists(probs_path):
        os.makedirs(probs_path)
    else:
        file_count_probs = len(
            [name for name in os.listdir(probs_path) if os.path.isfile(os.path.join(probs_path, name))])

    normal_path = os.path.join(datadir, 'DB_NORMAL_SEG')
    if not os.path.exists(normal_path):
        os.makedirs(normal_path)
    else:
        file_count_normal = len(
            [name for name in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, name))])

    if file_count_probs != 0 and file_count_normal != 0:
        return True
    else:
        return False

