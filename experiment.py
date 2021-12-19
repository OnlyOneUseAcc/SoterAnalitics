from tensorflow import keras as ks
import threading
import queue
import numpy as np
import pickle
import time
import json

from data_utils import get_sequence_len
from DataCollection import DataCollection


DATA_DIR = './data/'
DATA_FILE = 'data.pickle'
MODEL_PATH = 'models/best_model.h5'

N_SECONDS = 15
MAX_BUFFER_SIZE = 1000

# collection which imitate of tcp/ip 'post' request models from sensor/device
received_sensor_data = queue.Queue()

# model trained with param n_seconds == 15
model = ks.models.load_model(MODEL_PATH)

# dict to store model evaluation results
"""
# format - 
{(start, end) : class}
"""
model_results = {}

# Event to kill worker
pill2kill = threading.Event()


def model_worker(*args, **kwargs):
    last_n_seconds = DataCollection(size=get_sequence_len(N_SECONDS))
    while True:
        task = received_sensor_data.get()
        try:
            last_n_seconds.put(task)

            if last_n_seconds.size <= last_n_seconds.current_count:
                raw_data = last_n_seconds.get_data()
                predict = model.predict([
                    raw_data[:, 1].reshape(-1, get_sequence_len(N_SECONDS), 1),
                    raw_data[:, 2].reshape(-1, get_sequence_len(N_SECONDS), 1),
                    raw_data[:, 3].reshape(-1, get_sequence_len(N_SECONDS), 1)
                                         ])
                predict = np.argmax(predict, axis=1)
                model_results[str((raw_data[0, 0], raw_data[-1, 0]))] = str(predict[0])  # str for json serialization

                if len(model_results.keys()) >= MAX_BUFFER_SIZE:
                    # TODO clean up model_results
                    pass
            received_sensor_data.task_done()
        except:
            # TODO add exception handlers
            received_sensor_data.task_done()
            print('Worker can not handle task due to exception:')


if __name__ == '__main__':
    with open(f'{DATA_DIR}{DATA_FILE}', 'rb') as data_file:
        data = pickle.load(data_file)

    t = threading.Thread(target=model_worker, args=(pill2kill, "stop"), name='model evaluation', daemon=True)
    t.start()

    for index, sensor_data in enumerate(data):
        time.sleep(0.06)  # send data with frequency ~= 16hz
        received_sensor_data.put(sensor_data)

    received_sensor_data.join()
    pill2kill.set()
    with open(f'{DATA_DIR}experiment_results.json', 'w', encoding='UTF-8') as json_file:
        json.dump(model_results, json_file, indent=4, ensure_ascii=False)
