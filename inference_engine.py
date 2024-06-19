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
import json
from utils import timed_execution, log, generate_hash, nvidia
from singletons import AI, ProgramSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
from torch.cuda.amp import autocast
import spacy


def auto_detect_template(llm):
    """
    Attempt to auto detect the best template to use to save the user from shenanigans
    """
    ps = ProgramSettings()
    if not hasattr(auto_detect_template, "template_checked") or auto_detect_template.checked_model != llm.model_path:
        log("Attempting to detect templating style...")
        auto_detect_template.template_checked = True
        auto_detect_template.checked_model = llm.model_path
        config_path = os.path.join(llm.model_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="UTF-8") as tokenizer_config:
                data = json.load(tokenizer_config)
                if "chat_template" in data:
                    chat_template = data["chat_template"]
                else:
                    log("No template specification in tokenizer_config.json! Unable to auto detect! Will fall back to setting...")
                    auto_detect_template.template = ps.template
                    return auto_detect_template.template
            if "<|im_start|>" in chat_template:
                log("Detected template as 'Opus'!")
                auto_detect_template.template = "BAI Opus"
            elif "<|assistant|>" in chat_template:
                log("Detected template as 'Zephyr'!")
                auto_detect_template.template = "BAI Zephyr"
            elif "[INST]" in chat_template:
                log("Detected template as 'Instruct'!")
                auto_detect_template.template = "BAI Instruct"
            else:
                log("Unable to identify template! Will fall back to setting...")
                auto_detect_template.template = ps.template
        else:
            log("No tokenizer_config.json found in model! Will fall back to setting...")
            auto_detect_template.template = ps.template
            return auto_detect_template.template
    return auto_detect_template.template


def is_likely_code(text):
    """
    Simple heuristics to try to tell if the returned output is code, which needs different formatting

    Parameters:
    - text: A string containing the text to test

    Returns:
    - True if number of matches > threshold, else false
    """

    code_indicators = ["function", "var ", "{", "}", "import ", "#include", "def ", "class ", "->", "```"]
    threshold = 2  # Number of indicators to qualify as code
    return sum(indicator in text for indicator in code_indicators) >= threshold


@timed_execution
def load_model(new_model, queue):
    """
    Loads a new model to the user specified device configuration.
    Called as a thread, returns the model and tokenizer via queue

    Parameters:
    - new_model: The new model to load, a path to a directory
    - queue: The queue which we will return the new model and tokenizer through
    """
    torch.set_default_tensor_type("torch.cuda.FloatTensor" if nvidia() is True else "torch.FloatTensor")
    ps = ProgramSettings()
    log(f"Loading model {new_model}...")
    if os.path.exists(new_model):
        with torch.inference_mode():
            if ps.quant == "BNB 4bit":
                log("Quantizing model to 4-bit with BNB...")
                q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float32, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False)
                model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=q_config, low_cpu_mem_usage=True)
            elif ps.quant == "BNB 4bit+":
                log("Quantizing model to 4-bit+ with BNB...")
                q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=q_config, low_cpu_mem_usage=True)
            elif ps.quant == "BNB 8bit":
                log("Quantizing model to 8-bit with BNB...")
                q_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=5.0, llm_int8_enable_fp32_cpu_offload=True)
                model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=q_config, low_cpu_mem_usage=True)
            else:
                if ps.backend == "cuda":
                    model = AutoModelForCausalLM.from_pretrained(new_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
                elif ps.backend == "cpu":
                    model = AutoModelForCausalLM.from_pretrained(new_model, torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cpu")
                elif ps.backend == "auto":
                    model = AutoModelForCausalLM.from_pretrained(new_model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
            tokenizer = AutoTokenizer.from_pretrained(new_model, torch_dtype=torch.float16)
            streamer = TextStreamer(tokenizer)
        queue.put((model, tokenizer, streamer))
    else:
        log(f"Model path not found: {new_model}")
        ps.model_status = "unloaded"


@timed_execution
def threaded_model_response(llm, user_message, model_response, update_window):
    """
    This function is called via a thread and handles updating memory and working memory and then running inference

    Parameters:
        - llm: the llm we are working with
        - user_message: The new input as a string
        - model_response: a Queue to put the output in

    """
    ai = AI()
    with torch.inference_mode():
        # Update the AI's working memory dynamically based on context
        ai.working_memory = update_working_memory(user_message)
        now = str(datetime.now())
        identity = generate_hash(str(user_message) + str(now))
        ai.core_memory.append({"role": "user", "content": user_message, "identity": identity, "rating": "", "date": now})
        update_window.set()
        ai.working_memory.append({"role": "user", "content": user_message, "identity": identity, "rating": "", "date": now})
        # Run the language models and update the AI's memory with it's output
        response = generate_model_response(llm)
        now = str(datetime.now())
        identity = generate_hash(str(response) + str(now))
        ai.core_memory.append({"role": "assistant", "content": response, "identity": identity, "rating": "", "date": now})
        model_response.put(response)


def custom_template(llm):
    """
    Function to apply a custom template based on the program settings.
    """
    ai = AI()
    ps = ProgramSettings()
    cprompt = ""
    prompt = ai.working_memory
    if ps.auto_template is True:
        template = auto_detect_template(llm)
    else:
        template = ps.template
    if template == "HF Automatic":  # Use HF transformers build in apply_chat_template, doesn't always detect things properly
        cprompt = llm.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    elif template == "BAI Opus":
        for entry in ai.working_memory:
            if entry["role"] == "user":
                cprompt += f"<|im_start|>text names= {ps.username}\n"
                cprompt += entry["content"]
                cprompt += "<|im_end|>\n"
            elif entry["role"] == "assistant":
                cprompt += f"<|im_start|>text names= {ai.personality_definition['name']}\n"
                cprompt += entry["content"]
                cprompt += "<|im_end|>\n"
            elif entry["role"] == "system":
                cprompt += "<|im_start>|>system\n"
                cprompt += entry["content"]
                cprompt += "<|im_end|>\n"
        cprompt += f"<|im_start|>text names= {ai.personality_definition['name']}\n"
    elif template == "BAI SynthIA":
        for entry in prompt:
            cprompt += entry["role"].upper() + ":\n"
            cprompt += entry["content"] + "\n"
        cprompt += "ASSISTANT:\n"
    elif template == "BAI Instruct":
        for entry in ai.system_memory:
            cprompt += "[INST] "
            cprompt += entry["content"]
            cprompt += "[/INST]\n"
        for entry in ai.working_memory:
            if entry["role"] == "user":
                cprompt += "[INST] " + entry["content"] + "[/INST]"
            if entry["role"] == "assistant":
                cprompt += entry["content"] + "\n"
    elif template == "BAI Zephyr":
        for entry in prompt:
            cprompt += "<|" + entry["role"] + "|>\n"
            cprompt += entry["content"] + "\n"
        cprompt += "<|assistant|>"
    elif template == "BAI Alpaca":
        cprompt += "### Instruction: \n"
        for entry in ai.system_memory:
            cprompt += entry["content"] + "\n"
        cprompt += "### Input: \n"
        for entry in ai.working_memory:
            if entry["role"] == "user" and entry["content"] != "":
                cprompt += f"{ps.username}: "
                cprompt += entry["content"] + "\n"
            if entry["role"] == "assistant":
                cprompt += ai.personality_definition["name"] + ": "
                cprompt += entry["content"] + "\n"
        cprompt += "### Response: \n"
    log(f"Templated prompt: {cprompt}")
    return cprompt


def generate_model_response(llm):
    """
    This function cleans up the cuda context, prepares the prompt, runs the inference, post processes the output and returns the new output

    Returns:
    - The new output as a string
    """
    ai = AI()
    ps = ProgramSettings()
    # Clean up the context before inference
    if ps.backend in ["auto", "cuda"]:
        torch.cuda.empty_cache()
    gc.collect()
    cprompt = custom_template(llm)
    log("Tokenizing...")
    prompt_encoded = llm.tokenizer.encode(cprompt)  # This extra tokenization and decode step is necessary to keep track of where the model's response actually starts after tokens are munged by the tokenizer.
    prompt_decoded = llm.tokenizer.decode(prompt_encoded, skip_special_tokens=False)
    response_start = len(prompt_decoded)
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
        with autocast():
            outputs = llm.model.generate(prompt, max_new_tokens=ai.personality_definition["response_length"],
                                         streamer=llm.streamer if ps.do_stream and (ai.personality_definition["num_beams"] == 1) else None,
                                         do_sample=any([ai.personality_definition["temperature_enable"], ai.personality_definition["top_k_enable"], ai.personality_definition["top_p_enable"],
                                                        ai.personality_definition["typical_p_enable"], ai.personality_definition["repetition_penalty_enable"], ai.personality_definition["length_penalty_enable"]]),
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
    if ps.backend in ["auto", "cuda"]:
        torch.cuda.empty_cache()
    gc.collect()
    feedback = llm.tokenizer.decode(outputs[0], skip_special_tokens=False)
    return post_process(feedback, llm, response_start)


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
    # First we extract a list of keywords to be used for context matching. How many is set by the AI config.
    log("Updating working memory...")
    keywords = extract_keywords(user_message, ai.personality_definition)
    keywords_lower = [word.lower() for word in keywords]
    matching_memories = []

    # Always include the last stm_size entries - this is short term memory. Exclude system messages for now.
    recent_memories = [memory for memory in ai.core_memory[-ai.personality_definition["stm_size"]:] if memory.get("role") != "system"]

    # Search for keyword matches in core memory, excluding the most recent stm_size entries since they"re already included.
    for i, entry in enumerate(ai.core_memory[:-ai.personality_definition["stm_size"]]):
        if entry["role"] == "system":
            continue  # Skip system messages as they will be handled separately.
        content_lower = entry["content"].lower()
        if any(word in content_lower for word in keywords_lower):
            matching_memories.append(entry)
    # Reverse both of these so that the memories are in the correct temporal order
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
    # Select memories with a bias towards more recent matches and feedback.
    if matching_memories:
        # Create initial weights that increase linearly towards more recent entries.
        weights = [i**0.7 for i in range(1, len(matching_memories) + 1)]
        # Adjust weights based on matching_feedback
        for i, entry in enumerate(matching_memories):
            if entry["rating"] == "+":
                log("Upweighted: " + str(matching_memories[i]["content"]))  # Upvoted memories are MUCH more likely to be recalled.
                weights[i] *= 3
            elif entry["rating"] == "-":
                log("Downweighted: " + str(matching_memories[i]["content"]))  # Downvoted memories are half as likely to be recalled.
                weights[i] *= .25
        chosen_memories = weighted_selection(matching_memories, weights, min(ai.personality_definition["ltm_size"], len(matching_memories)))  # This picks the actual memories to use based on the weights
    else:
        chosen_memories = []
        log("No matching memories found.")

    selected_memories = chosen_memories + recent_memories + ai.guidance_messages
    for i, entry in enumerate(ai.guidance_messages):
        entry["turns"] -= 1
        if entry["turns"] == 0:
            log(f"Guidance message expired: {entry['content']} ")
            del ai.guidance_messages[i]
        else:
            ai.guidance_messages[i]["turns"] = entry["turns"]

    # Insert system messages at the beginning of the working memory.
    new_working_memory = ai.system_memory + selected_memories
    log(f"Working memory updated. Entries: {len(new_working_memory)}")
    log(f"New working memory: {new_working_memory}")
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


def post_process(input_string, llm, response_start):
    """
    Post processes the model output to prepare it for the user

    Parameters:
    - input_string:The string to process
    - llm: The language model for sniffing out the template

    Returns:
    - response: The processed string.

    """
    if len(input_string) > 0:
        ps = ProgramSettings()
        punctuation_set = {"!", ".", "?", "*", ")", '"'}
        if ps.auto_template is True:
            template = auto_detect_template(llm)
        else:
            template = ps.template
        end_tags = ["</s>"]
        if template == "HF Automatic":
            if input_string.find("<|im_end|>") != -1:
                applied_template = "Opus"
                end_tags = ["<|im_end|>"]
            elif input_string.find("<|assistant|>") != -1:
                applied_template = "Zephyr"
                end_tags = ["<|user|>", "<|system|>"]
            elif input_string.find("[INST]") != -1:
                applied_template = "Instruct"
                end_tags = ["[end of transmission]"]
            elif input_string.find("\nUSER:") != -1:
                applied_template = "Synthia"
                end_tags = ["\nUSER:"]
            log(f"Best guess template: {applied_template}")
        else:
            if template == "BAI SynthIA":
                applied_template = "Synthia"
                end_tags = ["\nUSER:"]
            elif template == "BAI Zephyr":
                applied_template = "Zephyr"
                end_tags = ["<|user|>", "<|system|>"]
            elif template == "BAI Opus":
                applied_template = "Opus"
                end_tags = ["<|im_end|>"]
            elif template == "BAI Instruct":
                applied_template = "Instruct"
                end_tags = ["[end of transmission]"]

        response = input_string[response_start:]
        for end_tag in end_tags:
            tag_index = response.find(end_tag)
            if tag_index != -1:
                response = response[:tag_index]
        if applied_template == "Zephyr":
            response = response.replace("<|assistant|>\n", "")
            response = response.replace("<|assistant|>", "")
        elif applied_template == "Opus":
            response = response.replace("<|im_start|>\n", "")
            response = response.replace("<|im_start|>", "")
        code = is_likely_code(response)
        log(f"Is likely code: {code}")
        if not code:
            if len(response) != 0 and response[-1] not in punctuation_set:
                last_punctuation_index = next((i for i, char in enumerate(
                    reversed(response)) if char in punctuation_set), None)
                if last_punctuation_index is not None:
                    result_string = response[:-last_punctuation_index].rstrip()
                    response = result_string
                response = response.strip()
            if len(response) != 0 and response[-1] == "*":
                # Last character is * so check the number of asterisks so we don't leave a dangling one
                if response.count("*") % 2 != 0:
                    response = response[:-1]
                    response = response.rstrip()
            elif response.count("*") % 2 != 0:
                response += "*"  # If the model didn't close it's roleplay, close it for her.
            response = response.replace("\n", "")
        else:
            # we need to find the last ``` and make sure truncation occurs after that
            pass
        response = response.strip()
    else:
        log("Response was blank")
        response = ""
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
