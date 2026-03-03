from flask import flask, request, render_template
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model = "llama3:latest")

def model(user_text):
    template = """context: {input_text}
    summarize the given context into 50 words, """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm 
    output = chain.invoke({"input_text": user_text})
    return output


app = flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    # Get form values and convert to float
    features = [x for x in request.form.values()]

    input_text = features[0]
    #print(input_text)

    output = model(input_text)

    return render_template('index.html', output_text = output)

if __name__ == "__main__":
    app.run(debug=True)


