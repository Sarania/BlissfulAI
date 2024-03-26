# -*- coding: utf-8 -*-
"""
Module containing components for inferencing with llms and processing their output for BlissfulAI
Created on Mon Mar 11 18:27:45 2024

@author: Blyss Sarania
"""
import random
import gc
import os
from datetime import datetime
from collections import OrderedDict
from utils import timed_execution, log, generate_hash, nvidia
from singletons import AI, LanguageModel, ProgramSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
import spacy

@timed_execution
def load_model(new_model, queue):
    """
    Loads a new model to the user specified device configuration. 
    Called as a thread, returns the model and tokenizer via queue
    
    Parameters:
    - new_model: The new model to load, a path to a directory
    - queue: The queue which we will return the new model and tokenizer through
    """
    ps = ProgramSettings()
    log("Loading model " + new_model + "...")
    if os.path.exists(new_model):
        with torch.inference_mode():
            if ps.quant=="BNB 4bit":
                log("Quantizing model to 4-bit with BNB...")
                q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False)
                model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=q_config, attn_implementation="sdpa", low_cpu_mem_usage=True)
            elif ps.quant=="BNB 4bit+":
                log("Quantizing model to 4-bit+ with BNB...")
                q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=q_config, attn_implementation="sdpa", low_cpu_mem_usage=True)
            elif ps.quant=="BNB 8bit":
                log("Quantizing model to 8-bit with BNB...")
                q_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=q_config, attn_implementation="sdpa", low_cpu_mem_usage=True)
            else:
                if ps.backend == "cuda":
                    model = AutoModelForCausalLM.from_pretrained(new_model, attn_implementation="sdpa", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
                elif ps.backend == "cpu":
                    model = AutoModelForCausalLM.from_pretrained(new_model, attn_implementation="sdpa", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cpu")
                elif ps.backend == "auto":
                    model = AutoModelForCausalLM.from_pretrained(new_model, attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)

            tokenizer = AutoTokenizer.from_pretrained(new_model, torch_dtype=torch.float16)
            streamer = TextStreamer(tokenizer)
        queue.put((model, tokenizer, streamer))
    else:
        log(f"Model path not found: {new_model}")
        ps.model_status = "unloaded"

@timed_execution
def threaded_model_response(user_message, model_response):
    """
    This function is called via a thread and handles updating memory and working memory and then running inference
    
    Parameters:
        - user_message: The new input as a string

    """
    ai = AI()
    with torch.inference_mode():
        # Update the AI's working memory dynamically based on context
        ai.working_memory = update_working_memory(user_message)
        now=str(datetime.now())
        identity=generate_hash(str(user_message)+str(now))
        ai.core_memory.append({"role": "user", "content": user_message, "identity": identity, "rating": "", "date": now})
        ai.working_memory.append({"role": "user", "content": user_message, "identity": identity, "rating": "", "date": now})

        # Run the language models and update the AI's memory with it's output
        response = generate_model_response()
        now=str(datetime.now())
        identity=generate_hash(str(response)+str(now))
        ai.core_memory.append({"role": "assistant", "content": response, "identity": identity, "rating": "", "date": now})
        model_response.put(response)

def custom_template():
    """
    Function to apply a custom template based on the program settings. 
    """
    ai = AI()
    ps = ProgramSettings()
    llm = LanguageModel()
    prompt=ai.working_memory
    log(ai.working_memory)
    cprompt = ""
    if ps.template == "HF Automatic": #Use HF transformers build in apply_chat_template, doesn't always detect things properly
        cprompt = llm.tokenizer.apply_chat_template(prompt, tokenize=False)
    elif ps.template =="BAI Opus":
        for entry in ai.working_memory:
            if entry["role"] == "user":
                cprompt+=f"<|IM_START|>text names= {ps.username}\n"
                cprompt+=entry["content"]
                cprompt+="<|IM_END|>\n"
            elif entry["role"] =="assistant":
                cprompt+=f"<|IM_START|>text names= {ai.personality_definition['name']}\n"
                cprompt+=entry["content"]
                cprompt+="<|IM_END|>\n"
            elif entry["role"] == "system":
                cprompt+="<|IM_START|>system\n"
                cprompt+=entry["content"]
                cprompt+="<|IM_END|>\n"
        cprompt+=f"<|IM_START|>text names= {ai.personality_definition['name']}\n"
    elif ps.template == "BAI Instruct":
        for entry in ai.system_memory:
            cprompt+="[INST] "
            cprompt+=entry["content"]
            cprompt+="[/INST]+\n"
        for entry in ai.working_memory:
            if entry["role"] == "user":
                cprompt+="<s>[INST] " + entry["content"] + "[/INST]"
            if entry["role"] == "assistant":
                cprompt+=entry["content"] + "</s>\n"
    elif ps.template =="BAI Zephyr":
        for entry in prompt:
            cprompt += "<|" + entry["role"] + "|>\n"
            cprompt += entry["content"] + "</s>\n"
        cprompt += "<|assistant|>"
    elif ps.template == "BAI Alpaca":
        cprompt += "### Instruction: \n"
        for entry in ai.system_memory:
            cprompt+= entry["content"] + "\n"
        cprompt+= "### Input: \n"
        for entry in ai.core_memory:
            if entry["role"] == "user" and entry["content"] != "":
                cprompt+= f"{ps.username}: "
                cprompt+=entry["content"] + "\n"
            if entry["role"] == "assistant":
                cprompt+= ai.personality_definition["name"] + ": "
                cprompt+=entry["content"] + "\n"
        cprompt+= "### Response: \n"
    log(cprompt)
    return cprompt

def generate_model_response():
    """
    This function cleans up the cuda context, prepares the prompt, runs the inference, post processes the output and returns the new output
    
    Returns:
    - The new output as a string
    """
    ai = AI()
    llm = LanguageModel()
    ps = ProgramSettings()
    # Clean up the context before inference
    if nvidia(): 
        if ps.backend in ["auto", "cuda"]:
            torch.cuda.empty_cache()
    gc.collect()
    cprompt=custom_template()
    #log(cprompt)
    log("Tokenizing...")
    if ps.backend in ("cuda", "auto"):
        log("Prompt to cuda...")
        prompt = llm.tokenizer.encode(cprompt, return_tensors="pt").to("cuda")
    elif ps.backend == "cpu":
        log("Prompt to cpu...")
        prompt = llm.tokenizer.encode(cprompt, return_tensors="pt").to("cpu")
    else:
        log(f"Prompt to {ps.backend}...")
        prompt = llm.tokenizer.encode(cprompt, return_tensors="pt").to("device")
    if any([ai.personality_definition["temperature_enable"], ai.personality_definition["top_k_enable"], ai.personality_definition["top_p_enable"], ai.personality_definition["typical_p_enable"], ai.personality_definition["repetition_penalty_enable"], ai.personality_definition["length_penalty_enable"]]):
        log("Using sampling...")
    else:
        log("Greedy!")
    log("Running primary model...")
    with torch.inference_mode():
        outputs = llm.model.generate(prompt, max_new_tokens=ai.personality_definition["response_length"],
                                 streamer=llm.streamer if ps.do_stream else None,
                                 do_sample=any([ai.personality_definition["temperature_enable"], ai.personality_definition["top_k_enable"], ai.personality_definition["top_p_enable"], ai.personality_definition["typical_p_enable"], ai.personality_definition["repetition_penalty_enable"], ai.personality_definition["length_penalty_enable"]]),
                                 temperature=ai.personality_definition["temperature"] if ai.personality_definition["temperature_enable"] else None,
                                 num_beams=ai.personality_definition["num_beams"],
                                 top_k=ai.personality_definition["top_k"] if ai.personality_definition["top_k_enable"] else None,
                                 top_p=ai.personality_definition["top_p"] if ai.personality_definition["top_p_enable"] else None,
                                 typical_p=ai.personality_definition["typical_p"] if ai.personality_definition["typical_p_enable"] else None,
                                 length_penalty=ai.personality_definition["length_penalty"] if ai.personality_definition["length_penalty_enable"] else None,
                                 repetition_penalty=ai.personality_definition["repetition_penalty"] if ai.personality_definition["repetition_penalty_enable"] else None,
                                 pad_token_id=llm.tokenizer.eos_token_id)
    # Clean up the context after inference
    prompt = None
    del prompt
    if nvidia():
        if ps.backend in ["auto", "cuda"]:
            torch.cuda.empty_cache()
    gc.collect()
    feedback = llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_process(feedback)

@timed_execution
def update_working_memory(user_message):
    """
    Updates the AI's working memory based on the user's input and core memory contents.
    It filters core memories relevant to the current user message using keyword matching, 
    prioritizing recent entries and system messages.

    Parameters:
    - user_message: The user"s input, a string
 
    Returns:
    - a list of dictionaries to function as the working memory
    """
    ai = AI()
    #First we extract a list of keywords to be used for context matching. How many is set by the AI config.
    log("Updating working memory...")
    keywords = extract_keywords(user_message, ai.personality_definition)
    keywords_lower = [word.lower() for word in keywords]
    matching_memories = [] # these are

    # Always include the last stm_size entries - this is short term memory. Exclude system messages for now.
    recent_memories = [memory for memory in ai.core_memory[-ai.personality_definition["stm_size"]:] if memory.get("role") != "system"]


    # Search for keyword matches in core memory, excluding the most recent stm_size entries since they"re already included.
    for i, entry in enumerate(ai.core_memory[:-ai.personality_definition["stm_size"]]):
        if entry["role"] == "system":
            continue  # Skip system messages as they will be handled separately.
        content_lower = entry["content"].lower()
        if any(word in content_lower for word in keywords_lower):
            matching_memories.append(entry)
    #Reverse both of these so that the memories are in the correct temporal order
    matching_memories = reversed(matching_memories)
    # Create an OrderedDict to preserve order while removing duplicates
    unique_entries = OrderedDict()

    # Iterate through matching_memories while preserving order
    for entry in matching_memories:
        content = entry["content"]
        # Add the entry to the OrderedDict only if it"s not already present
        if content not in unique_entries:
            unique_entries[content] = entry

    # Extract the values from the OrderedDict to maintain the order
    matching_memories = list(unique_entries.values())
    #log(matching_memories)
    # Select memories with a bias towards more recent matches and feedback.
    if matching_memories:
        # Create initial weights that increase linearly towards more recent entries.
        weights = [i**0.7 for i in range(1, len(matching_memories) + 1)]
        # Adjust weights based on matching_feedback
        for i, entry in enumerate(matching_memories):
            if entry["rating"] == "+":
                log("Upweighted: " + str(matching_memories[i]["content"])) #Upvoted memories are MUCH more likely to be recalled.
                weights[i] *= 3
            elif entry["rating"] == "-":
                log("Downweighted: " + str(matching_memories[i]["content"]))#Downvoted memories are half as likely to be recalled.
                weights[i] *= .25
        chosen_memories = weighted_selection(matching_memories, weights, min(ai.personality_definition["ltm_size"], len(matching_memories))) #This picks the actual memories to use based on the weights
    else:
        chosen_memories = []
        log("No matching memories found.")

    # Combine selected matches with recent and system messages, ensuring uniqueness.
    selected_memories = chosen_memories + recent_memories

    # Insert system messages at the beginning of the working memory.
    new_working_memory = ai.system_memory + selected_memories
    log(f"Working memory updated. Entries: {len(new_working_memory)}")
    log(new_working_memory)
    return new_working_memory


def extract_keywords(sentence, personality_definition):
    """
    Intelligently extracts the relevant context keywords from the input using a small language model

    Parameters:
    - sentence: The sentence to extract keywords from, a string
    - personality_definition: The personality definition settings so we can know how many keywords we need

    Returns:
    - keywords: A list of keywords

    """
    # Load the English language model
    nlp = spacy.load("en_core_web_trf")
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Initialize a list for meaningful lemmas and a set for seen lemmas to avoid duplicates
    meaningful_lemmas = []
    seen_lemmas = set()

    # Define priority order for parts of speech
    parts_of_speech_priority = ["PROPN", "NOUN", "VERB", "ADJ"]

    # Try to collect lemmas according to the priority of parts of speech
    for pos in parts_of_speech_priority:
        for token in doc:
            lemma = token.lemma_.lower()  # Use lower case to ensure case-insensitive matching
            # Check if current part of speech matches, the lemma is not seen yet, and we need more lemmas
            if token.pos_ == pos and lemma not in seen_lemmas and len(meaningful_lemmas) < personality_definition["num_keywords"]:
                meaningful_lemmas.append(lemma)
                seen_lemmas.add(lemma)
            # Break early if we have collected enough lemmas
            if len(meaningful_lemmas) == personality_definition["num_keywords"]:
                break
        if len(meaningful_lemmas) == personality_definition["num_keywords"]:
            break

    # Take the first three unique meaningful lemmas, now potentially including base forms of verbs and adjectives
    keywords = meaningful_lemmas[:personality_definition["num_keywords"]]

    log("Extracted keywords: " + str(keywords))
    return keywords

def post_process(input_string):
    """
    Post processes the model output to prepare it for the user
    
    Parameters:
    - input_string:The string to process

    Returns:
    - response: The processed string.

    """
    ai = AI()
    ps = ProgramSettings()
    punctuation_set = {"!", ".", "?", "*", ")"}
    #log("Model output: " + str(input_string))
    #We assume the response starts from the last occurence of <|assistant|>
    #This might not be true for all models and can break if the model outputs weird shit!
    if ps.template != "BAI Opus":
        response_start = input_string.rfind("<|assistant|>")
        response_start += 13
        response = input_string[response_start:]
        #Here we truncate the output to the last occurence of specific punctuation
        if len(response) != 0 and response[-1] not in punctuation_set:
            last_punctuation_index = next((i for i, char in enumerate(
                reversed(response)) if char in punctuation_set), None)
            if last_punctuation_index is not None:
                result_string = response[:-last_punctuation_index].rstrip()
                response = result_string
        response = response.strip()
        if response[-1] == "*":
            #Check the number of asterisks so we don"t leave a dangling one
            if response.count("*") % 2 != 0:
                response = response[:-1]
                response = response.rstrip()
    elif ps.template == "BAI Opus":
        fstring = f"{ai.working_memory[-2]['content']}<|IM_END|>\n<|IM_START|>text names= {ps.username}\n{ai.working_memory[-1]['content']}<|IM_END|>\n<|IM_START|>text names= {ai.personality_definition['name']}\n"
        log(fstring)
        response_start = input_string.rfind(fstring) + len(fstring)
        response = input_string[response_start:]
        log(response)
        #newline_pos = response.find("\n") + 1  # Adding 1 to start after the "\n"
        #response = response[newline_pos:]
        tag_index=response.find("<|IM_END|>")
        if tag_index != -1:
            response=response[:tag_index]
        else:
            if len(response) != 0 and response[-1] not in punctuation_set:
                last_punctuation_index = next((i for i, char in enumerate(
                    reversed(response)) if char in punctuation_set), None)
                if last_punctuation_index is not None:
                    result_string = response[:-last_punctuation_index].rstrip()
                    response = result_string
                    response = response.strip()
            if response[-1] == "*":
                #Check the number of asterisks so we don"t leave a dangling one
                if response.count("*") % 2 != 0:
                    response = response[:-1]
                    response = response.rstrip()
        response = response.strip()
    #log("Response is " + len(response.split()) + " words in length.")
    return response

def weighted_selection(keyword_memories, weights, max_length):
    """
    Without selecting duplicates, extracts from memory according to weights
    
    Parameters:
    - keyword_memories: The memories that matched our keyword search, a list of dictionaries
    - weights: The weights to use, biasing more towards recent memories and also affected by feedback
    - max_length: The max length to use
    """
    selected_items = []
    for _ in range(min(max_length, len(keyword_memories))):
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Choose an item based on normalized weights
        chosen_index = random.choices(range(len(keyword_memories)), weights=normalized_weights, k=1)[0]
        selected_items.append(keyword_memories[chosen_index])
        # Remove the chosen item and its weight to avoid repetition
        del keyword_memories[chosen_index]
        del weights[chosen_index]

    return selected_items
