console.log("Hare Krishna")

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import * as dotenv from "dotenv"

dotenv.config()

import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { OpenAI,OpenAIEmbeddings } from "@langchain/openai";

import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

import { RetrievalQAChain } from "langchain/chains";

const loader = new PDFLoader("src/langchain.pdf");

const docs = await loader.load();

//splitter function
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
const question="give code for extracting from pdf using document loader "

const ans=await chain.call({
  query:question
})

console.log(ans)