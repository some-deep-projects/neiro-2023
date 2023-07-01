# LLM-based chatbot

## Solution Overview

1. This chatbot is built on a pretrained dialogue model which has been adapted through prompting, bypassing the need for additional fine-tuning. 
2. It leverages the ```transformers``` library and primarily utilizes the [pygmalion-6b](https://huggingface.co/PygmalionAI/pygmalion-6b) model. This model is a fine-tuned version of [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) on dialogue data. 
3. Dialog state is consistently maintained throughout the conversation using ```past_key_values```, thus facilitating smooth operation on a single T4 GPU without re-computing previous tokens or data. 
4. The decoding process is implemented progressively, ensuring compatibility with byte-level tokenizers.
5. The model generation process is halted on encountering predefined stop words stored in a *trie* data structure (```llm_chat.stop_criteria.WordStopCriterion```). 

## How the Run the Model

1. Clone the repository: ```git clone git@github.com:some-deep-projects/neiro-2023.git```
2. Navigate to the cloned repository: ```cd neiro-2023```
3. Build the docker image: ```docker build -t neiro .```
4. Run the docker image: ```docker run --rm -it neiro```

## Experimentation Insights

1. Purely prompted LLMs such as [Open-LLaMA-13B](https://huggingface.co/openlm-research/open_llama_13b) didn't yield satisfactory results.
2. Instructional models like [WizardLM 7B](https://huggingface.co/TheBloke/wizardLM-7B-GGML) also fell short in performance, consistently asking *how can I help you?*.
3. The [llama.cpp](https://github.com/ggerganov/llama.cpp) library proved to be very promising, however, the ```transformers``` library was chosen for its extendability.
