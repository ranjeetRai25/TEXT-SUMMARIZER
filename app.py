from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates  # UI e.g-- index.html ko integrate krne ke liye server side se
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# intialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer= T5Tokenizer.from_pretrained("./saved_summary_model")

# device
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory = ".")

# Input Schema for dialogue ==> strings
class DialogueInput(BaseModel):
    dialogue: str

# Data preprocessing 
def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text)  # html tags <p> <h1>
    text = text.strip().lower()  # remove extra spaces from start n end and convert all text into lowercase
    return text  


def summarize_dialogue(dialogue : str) -> str:

    # clean data
    dialogue = clean_data(dialogue)

    # tokenize
    inputs = tokenizer(
        dialogue, 
        padding = "max_length", 
        max_length = 512, 
        truncation = True,
        return_tensors = "pt"  # convert tokens in tensors, pt--> pytorch tensors
    )

    #  generate the summary ==> token ids
    target = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,   # In Ai there is one algorithm which is beam search, which will generate 4 outputs, and give best out of it
        early_stopping = True
        
    )

    # token ids convert to summary 
    # decoded our output 
    summary = tokenizer.decode(target[0], skip_special_tokens = True)   # EOS, seperator e.g <html> space 
    return summary


# API Endpoints
@app.post("/summarize/")    # end point /Summarize/
async def summarize(dialogue_input: DialogueInput):   # async ---> 3 task work parallely, sync--> 3 task done one by one
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )
