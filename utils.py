import logging
from logging.handlers import RotatingFileHandler
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import dotenv_values

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import DataFrameLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter



if __name__ == '__main__':
    print('please run main module')


def load_env():
    env = dotenv_values()
    os.environ['OPENAI_API_KEY'] = env.get('OPENAI_TOKEN') # OpenAI Key
    return env

env = load_env()


def create_logger(name, level):
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not os.path.exists(f'.\logs'):
        os.makedirs(f'.\logs')

    file_handler = RotatingFileHandler(filename=f'logs\{name}.log',
                                       maxBytes=1*1024*1024,
                                       backupCount=10,
                                       encoding='utf-8')

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if level == 'DEBUG' else logging.ERROR)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_patient_data(patient_id):
    with open(f'./data/{patient_id}/data.json', 'r') as f:
        data = json.load(f)

    # load data into into pandas dataframe
    df = pd.DataFrame.from_dict(data)

    age = int(df['Age'].mean())
    avg_weight = int(df['AvgWeight'].mean())
    avg_height = int(df['AvgHeight'].mean())

    avg_noise = int(df['AvgNoise'].mean())
    avg_movements = int(df['AvgMovements'].mean())
    avg_heart_rate = int(df['AvgHeartRate'].mean())
    avg_temperature = int(df['AvgTemperature'].mean())
    avg_oxygen_saturation = int(df['AvgOxygenSaturation'].mean())

    patient_data = {'PatientID': patient_id,
                    'PatientName': df['PatientName'][0],
            'Age': age,
            'AvgWeight': avg_weight,
            'AvgHeight': avg_height,
            'AvgNoise': avg_noise,
            'AvgMovements': avg_movements,
            'AvgHeartRate': avg_heart_rate,
            'AvgTemperature': avg_temperature,
            'AvgOxygenSaturation': avg_oxygen_saturation}

    for col in [c for c in df.columns if c not in ['PatientID', 'PatientName', 'SampleID', 'Timestamp', 'Age']]:
        update_patient_plots(df, column=col)


    return patient_data


def update_patient_plots(df, column):
    patient_id = df.PatientID[0]
    sns.lineplot(data=df, x="Timestamp", y=column)
    # add patient id to the plot
    plt.title(f'Patient ID - {patient_id}')
    # set reference line at mean + 1 std
    plt.axhline(y=df[column].mean() + df[column].std(), color='r', linestyle='--')

    if not os.path.exists(f'./plots/{patient_id}'):
        os.makedirs(f'./plots/{patient_id}')

    plt.savefig(f'./plots/{patient_id}/{column}.png')
    plt.clf()


def validate_telegram_user_id(user_id):
    """
    Checks if the user id in the white list.

    Args:
        user_id (int): The user id to check.

    Returns:
        bool: True if the user id is valid, False otherwise.
    """
    return user_id in json.loads(env.get('TELEGRAM_USER_IDS'))


# def convert_ogg_to_mp3(ogg_file_path, mp3_file_path):
#     """
#     Converts a .ogg file to .mp3 file.

#     Args:
#         ogg_file_path (str): The path to the .ogg file.
#         mp3_file_path (str): The path to the .mp3 file.

#     Returns:
#         None
#     """
#     # Add FFmpeg to the system's PATH
#     # os.environ["PATH"] += os.pathsep + 'C:/Users/user/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin'  # Replace '/path/to/ffmpeg/bin' with the actual path
#     sound = AudioSegment.from_ogg(ogg_file_path)
#     sound.export(mp3_file_path, format="mp3")


# os.path.join('.', 'test1.mp3')

def create_or_get_vector_store(chunks: list, index_name: str) -> FAISS:
    """
    Create or get vector store

    Args:
        chunks (list): List of chunks

    Returns:
        FAISS: Vector store
    """
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings() # if you want to use open source embeddings

    if not os.path.exists(f"./db/{index_name}.faiss"):
        print("CREATING DB")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(folder_path="./db", index_name=index_name)
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local(folder_path="./db", embeddings=embeddings, index_name=index_name)

    return vectorstore


