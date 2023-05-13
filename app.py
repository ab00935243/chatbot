from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json['input_text']
    response = chatbot(input_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    index = construct_index("docs")
    app.run()
