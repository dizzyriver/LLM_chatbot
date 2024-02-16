# Skills_Edge_Assistant

LLM chatbot using Retrieval Augmented Generation with [Langchain](https://www.langchain.com/), [Gradio](https://www.gradio.app/) and [Qrant](https://qdrant.tech/).

 * Install Qdrant. Working with the [Docker installation](https://qdrant.tech/documentation/quick-start/) is straightforward.
 * Create a conda environment with `conda env create -f ai_env.yml`
 * For GPU support on macos follow [these instruction](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/) to install `llama-cpp-python`
 * Download LM studio [LM Studio](https://lmstudio.ai/) which gives you access to open source llm's from [huggingface](https://huggingface.co/.)
     * In the search area find and download a model of your choosing.
     * We have developed and tested the chatbot against the Mistral model `mistral-7b-instruct-v0.2.Q6_K` and it performs well.
     * Once the model has been downloaded start the model server in LM studio.
 * Rename .env.example to .env and change the `PROJECT_HOME` variable.
 * Create a folder named `documents` under `PROJECT_HOME`. Contact `vasilis.hatzopoulos@mercer.com` to receive the Skills Edge documents. The application will also run with any other set of `pdf` or `pptx` documents, you just need to place them in a folder under `documents` and point to it in the `DOCUMENT_PATH` environment variable.
 * Launch Qdrant
 * Executing all cells in the notebook `Skills-Edge-Assistant_v02.ipynb` will run the RAG pipeline and start a chatbot on `localhost`
 * Model and RAG parameters are controlled via the file `config/rag_conf.yaml`. A variety of [RAG techniques](https://python.langchain.com/docs/modules/data_connection/retrievers/) are supported and can be turned on or off from this configuration file
 * For Skills Edge we have a custom prompt in `prompts/skill_edge_prompt.txt`. Feel free to create other prompts for your own domain.
 
 
