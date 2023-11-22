import os
import glob
import logging
import re
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = 'data'
REGEX = r"(?P<surgery>(pre|post))/(?P<filename>(?P<patient>[0-9A-Z]{8})_([0-9A-Z]{8}_)+(?P<side>[a-z]+)(_right)*.jpg)"
REGEX = os.path.join(DATA_DIR, REGEX)


if __name__ == '__main__':
    paths = glob.glob(f'{DATA_DIR}/**/*.jpg', recursive=True)

    data_dict_list = []
    for path in paths:
        data_dict = {'path': path}
        data_dict.update(**re.match(REGEX, path).groupdict())
        data_dict_list.append(data_dict)

        logger.info(data_dict['path'])
        logger.info(data_dict['filename'])
        logger.info(data_dict['surgery'])
        logger.info(data_dict['patient'])
        logger.info(data_dict['side'])

    df = pd.DataFrame(data_dict_list)
    df.to_csv(os.path.join(DATA_DIR, 'data.csv'), index=False)
