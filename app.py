import torch
from fastapi import FastAPI
import uvicorn
from transformers import RobertaForSequenceClassification, AutoTokenizer
import mains.classifier as clf

# Data structure
from schema.item import Item

#define the fastapi
app = FastAPI(title="Sentiment Analysis API",
            description="API for Sentiment Analysis",
            version="1.0")


#when the app start, load the model
@app.on_event('startup')
async def load_model():
    clf.model = RobertaForSequenceClassification.from_pretrained("./models/phobert_base_5epochs", local_files_only = True)
    clf.tokenizer = AutoTokenizer.from_pretrained("./models/phobert_base_5epochs", local_files_only = True)
    
    
#when post event happens to /predict
@app.post('/api/v1/sentiment')
async def get_prediction(item:Item):
    input_ids = torch.tensor([clf.tokenizer.encode(item.raw)])
    output = clf.model(input_ids)
    item.sentiment = output.logits.softmax(dim=-1).tolist()[0]
    return {'negative': item.sentiment[0],
            'positive': item.sentiment[1],
            'neutral': item.sentiment[2]}

if __name__ == '__main__':
    uvicorn.run(app, port=5000, host='localhost')