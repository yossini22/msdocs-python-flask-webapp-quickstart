from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
                                ChatPromptTemplate,
                                MessagesPlaceholder,
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                PromptTemplate
                                )
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import os

from openai import OpenAI
from utils import create_or_get_vector_store




if __name__ == '__main__':
    print('please run main module')


def init_memory():
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key="user_message")

    return memory


def asistant_router(user_message, logger):
    logger.info(f'start asistant router')
    class give_advice(BaseModel):
        """Call this to give an advice, answer, asist about a patient question"""

    class send_graph(BaseModel):
        """Call this when asked to send a graph, trend, plot, statistics about the data generatied or geathered from the sensors.
        The data can be the folowing:
        - noise level
        - nuber of movements
        - heart rate
        - temperature
        - oxygen saturation
        """
        graph_names: List = Field(description="the data that should be on the desiered graph. The data can be the folowing: AvgWeight, AvgHeight, AvgNoise, AvgMovements, AvgHeartRate, AvgTemperature, AvgOxygenSaturation. If none of the above is given, the default graph is AvgHeartRate. If asked for more than one graph, give the names in array format like 'name1', 'name2',.... if asked for all graphs give 'all' as the answer.")

    # class analyze_sensor_data(BaseModel):
    #     """Call this when asked to analyze the data generatied or geathered from the sensors."""

    class default_route(BaseModel):
        """Call this as a default function if you dont know what other function to call"""

    functions = [
        convert_pydantic_to_openai_function(give_advice),
        convert_pydantic_to_openai_function(send_graph),
        # convert_pydantic_to_openai_function(analyze_sensor_data),
        convert_pydantic_to_openai_function(default_route)
    ]

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """You are a AI assistant that helps to choose the right tool to run according to the user message.
                This is a function call and you can choose only one from the list of functions provided.
                
                The user message can be about the patient or child or baby and related to health, growth, development, parental instruction, sensor data.
                
                If you dont know what function to choose, choose "default_route" as a default function.
                """
            ),
            HumanMessagePromptTemplate.from_template("{user_message}")
        ]
    )

    model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-1106')
    model_with_function = model.bind(functions=functions)
    chain = prompt | model_with_function | JsonOutputFunctionsParser(args_only=False)
    result = chain.invoke(input={"user_message": user_message})

    logger.info(f'asistant_router result - {result}')

    return result


def give_advice(user_message, patient_data_dict, memory, logger):
    # logger.info(f'start give advice')
    # """
    # Generates a response using the ChatOpenAI model for the given message.

    # Args:
    #     message (str): The input message for the model.

    # Returns:
    #     str: The generated response from the model.
    # """

    # prompt = ChatPromptTemplate(
    #     messages=[
    #         SystemMessagePromptTemplate.from_template(
    #             """Act as a senior lactation and sleep consultant, highly expert in your field. You have extensive experience in personally advising parents of infants, providing reliable and practical advice in clear simple language.

    #             You address parents respectfully and without judgment. Provide extensive useful information based on credible sources and personal experience.

    #             Answer questions clearly, concisely and in a friendly manner. Avoid using complex medical terminology.

    #             Eager to assist parents and make this challenging period easier for them as much as possible.

    #             ---------------------------------------------------
 
    #             If the question is not clear, ask for more details but dont answer if you dont know the right answer. 
                
    #             Your answer should be short and clear, with personal attention to childs name
                
    #             Your answer should be relevant to the specific child with the folowing information:
    #             Patient ID: {PatientID}
    #             Patient Name: {PatientName}
    #             Patient Age: {Age} months
    #             Patient Height: {AvgHeight} cm
    #             Patient Weight: {AvgWeight} kg

    #             Additional information from the last night sensors data monitoring:
    #             Average Noise: {AvgNoise}
    #             Average Movements: {AvgMovements}
    #             Average Heartrate: {AvgHeartRate}
    #             Average Temperature: {AvgTemperature}
    #             Average Oxygen Saturation: {AvgOxygenSaturation}
    #             """
    #         ),
    #         MessagesPlaceholder(variable_name="chat_history"),
    #         HumanMessagePromptTemplate.from_template("{user_message}")
    #     ]
    # )

    # model = ChatOpenAI(temperature=0.2, model='gpt-3.5-turbo-1106')

    # conversation = LLMChain(llm=model,
    #                         prompt=prompt,
    #                         verbose=True,
    #                         memory=memory
    #                         )

    # result = conversation.invoke(input={"user_message": user_message,
    #                               **patient_data_dict})


    # logger.info(f'give_advice result - {result.get("text")}')

    result = {'text': rag(user_message)}

    return result.get('text')


def voice_to_text(voice_file_path, logger):
    from openai import OpenAI
    client = OpenAI()

    with open(f"{voice_file_path}", "rb") as audio_file:
        trascribe = client.audio.transcriptions.create(model="whisper-1", # transcriptions
                                                        file=audio_file,
                                                        response_format="json",
                                                        )

    with open(f"{voice_file_path}", "rb") as audio_file:
        translate = client.audio.translations.create(model="whisper-1", # transcriptions
                                                        file=audio_file,
                                                        response_format="json",
                                                        )

    text_org = trascribe.text
    text_eng = translate.text
    logger.info(f'voice_to_text trascribe - {text_org}')
    logger.info(f'voice_to_text translate - {text_eng}')

    return text_org, text_eng


# --------------------------------------------------






def rag(user_question):
    # ----------------------------- -------------------------------------------------------------------------
    # qa documents with qa messages memory
  
    # prompt = ChatPromptTemplate(
    #     messages=[SystemMessagePromptTemplate.from_template(
    #     """
    #     You are a chatbot tasked with responding to questions about the Children and parents.

    #     You should never answer a question with a question, and you should always respond with 2 most relevant pages.

    #     Do not answer questions that are not about the LangChain library or project.

    #     Given a question, you should respond with the 2 most relevant pages by following the relevant context below:\n
    #     {context}
    #     """),
    #     HumanMessagePromptTemplate.from_template("{question}")
    #     ]
    # )

    # vector_store = create_or_get_vector_store(['abc'], index_name="balaolam")
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    # retriever=vector_store.as_retriever()
    # # llm = HuggingFaceHub(model="HuggingFaceH4/zephyr-7b-beta") # if you want to use open source LLMs
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     memory=memory,
    #     combine_docs_chain_kwargs={
    #         "prompt": prompt
    #     },
    # )
    # response = conversation_chain({"question": user_question})
    # answer = response[('answer')]
    # # ----------------------------- -------------------------------------------------------------------------
    # # simple qa documents

    vector_store = create_or_get_vector_store(['abc'], index_name="all_files_from_google_drive")
    vector_store2 = create_or_get_vector_store(['abc'], index_name="all_urls_from_adi_word")
    vector_store.merge_from(vector_store2)
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
                                    llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True#,
                                    # chain_type_kwargs={"prompt": prompt}
                                    )
    
    # query = "How many AI publications in 2021?"
    result = qa({"query": user_question})

    # print(result['query'])
    # print(result['result'])

    
    
        


    # ----------------------------- -------------------------------------------------------------------------
    # simple load qa chain on documents

    # vector_store = create_or_get_vector_store(['abc'], index_name="all_files_from_google_drive")
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    # retriever=vector_store.as_retriever(search_kwargs={"k": 5})
    # retrieved_docs = retriever.get_relevant_documents(user_question)
    # qa = load_qa_chain(
    #                     llm=llm, 
    #                     chain_type="stuff"
    #                     )
    
    # result = qa.run(input_documents=retrieved_docs, question = user_question)    

    answer = result['result'] + "\n\n\n" + "----------" "\n" + 'sources:\n' + '\n'.join([document.metadata['source'] for document in result['source_documents']])
    return answer


# , 'chat_history':'aaaa'