console.log("Hare Krishna")

import express from 'express';

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import * as dotenv from "dotenv"

dotenv.config()

import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { OpenAI,OpenAIEmbeddings } from "@langchain/openai";

import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

import { RetrievalQAChain } from "langchain/chains";

const loader = new PDFLoader("src/langchain.pdf");

const docs = await loader.load();

const app = express();

const port = 3000; 


  

app.get('/:data', async (req, res) => {

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 20,
  });
  
  //splitted chunks
  const splittedDocs = await splitter.splitDocuments(docs);
  
  const openAIApiKey = "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"
  const embeddings = new OpenAIEmbeddings()
  
    const vectorStore = await HNSWLib.fromDocuments(
     splittedDocs,
     embeddings
    );
  
   
  // console.log(splittedDocs)
  
  const vectorStoreRetriever = vectorStore.asRetriever();
  
  
  const model = new OpenAI({
      modelName:'gpt-3.5-turbo'
  });
  
  const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever);
  
  // console.log(chain)
  const question=req.params.data
  
  const ans=await chain.call({
    query:question
  })
  

  res.send(ans);
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

