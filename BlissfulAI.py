# -*- coding: utf-8 -*-
"""
BlissfulAI

Description:
BlissfulAI is a program designed to harness the power of Large Language Models (llms).
The primary goal is to leverage advanced natural language processing capabilities provided by llms to build a sophisticated AI system
capable of understanding, generating, and responding to human-like language as well as other modalities.

Dependencies:
- Python 3.x and associated libraries ("pip install -r requirements.txt")
- Large Language Model (llm) library (e.g., Zephyr7b, OpenChat)

Note:
BlissfulAI is a work in progress, and regular updates may enhance its capabilities and performance. Refer to the documentation for detailed
information on usage, configuration, and updates.

Disclaimer:
This program is developed for educational and experimental purposes. Use with caution, and ensure compliance with ethical guidelines and legal
regulations related to AI development and deployment. ChatGPT wrote this disclaimer, Blyss wrote this description XD
Created on Mon Mar 4 12:00:00 2024

@author: Blyss Sarania
"""
import signal
import os
import sys
import gc
import json
import argparse
import threading
import shutil
import warnings
import tkinter as tk
from datetime import datetime
from queue import Queue
import webbrowser
import PySimpleGUI as sg
from inference_engine import threaded_model_response, threaded_story_response, load_model
from utils import log, timed_execution, is_number, update_system_status, animate_ellipsis, generate_hash, get_cpu_name, get_gpu_info, get_ram_usage, get_os_name_and_version, nvidia
from singletons import AI, ProgramSettings
import torch
if sys.platform == "win32":
    import win32api
    import win32con


class LanguageModel():
    """
    This class manages the language model, its path, tokenizer, and an optional streaming interface.

    Attributes:
        _model: Stores the actual language model object.
        _model_path: Stores the path to the language model file.
        _tokenizer: Stores the tokenizer associated with the language model.
        _streamer: Optionally stores a streaming interface for the language model.
    """

    def __init__(self):
        """
        Initializes the language_model instance with default values.
        """
        self._model = None
        self._model_path = None
        self._tokenizer = None
        self._streamer = None

    @property
    def model(self):
        """
        Gets the current language model object.

        Returns:
            The language model object if it exists, otherwise None.
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Sets the language model object, ensuring it has a forward method.

        Args:
            value: The new language model object to set.

        Raises:
            ValueError: If the provided model does not have a forward method.
        """
        if value is not None and not hasattr(value, "forward"):
            raise ValueError("model must be a model with a forward method")
        self._model = value

    @property
    def model_path(self):
        """
        Gets the current path to the language model file.

        Returns:
            The path to the language model file as a string, or None if not set.
        """
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        """
        Sets the path to the language model file.

        Args:
            value: The new path to the language model file as a string.

        Raises:
            ValueError: If the provided value is not a string.
        """
        if not isinstance(value, str) and value is not None:
            raise ValueError("model_path must be a string")
        self._model_path = value

    @property
    def tokenizer(self):
        """
        Gets the tokenizer associated with the language model.

        Returns:
            The tokenizer object if it exists, otherwise None.
        """
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        Sets the tokenizer associated with the language model, ensuring it has an encode method.

        Args:
            value: The new tokenizer object to set.

        Raises:
            ValueError: If the provided tokenizer does not have an encode method.
        """
        if value is not None and not hasattr(value, "encode"):
            raise ValueError("tokenizer must have an encode method")
        self._tokenizer = value

    @property
    def streamer(self):
        """
        Gets the streaming interface associated with the language model.

        Returns:
            The streaming interface object if it exists, otherwise None.
        """
        return self._streamer

    @streamer.setter
    def streamer(self, value):
        """
        Sets the streaming interface for the language model. The method does not enforce
        any specific requirements on the streamer object, allowing for flexible implementations.

        Args:
            value: The new streaming interface object to set.
        """
        self._streamer = value


def handle_memory_failed():
    """
    Displays a notice to the user that memory check failed and asks what to do

    Parameters:
    - message: The notice to display

    """
    layout = [
        [sg.Text("Detected errors in memory fingerprint. What would you like to do?")],
        [sg.Column([[sg.Button("Fix"), sg.Button("Ignore")]], justification="center")]
    ]
    window = sg.Window("Memory Check Failed!", layout, icon="./resources/bai.ico", modal=True)
    while True:
        event, _ = window.read()
        if event in (sg.WIN_CLOSED, "Ignore"):
            window.close()
            return False
        if event == "Fix":
            window.close()
            return True


def open_url(url):
    """
    Simply opens a URL in the default browser
    """
    webbrowser.open(url)


def graceful_shutdown():
    """
    clean things up and shut it down
    """
    ps = ProgramSettings()
    ai = AI()
    if ps.model_status == "inferencing":
        log("Force quit confirmed. Removing last user_message to keep memory consistent.")
        ai.core_memory.pop()
    log("Final save of personality...")
    update_hard_memory()
    log("Exiting...")
    sys.exit(0)


def handle_ccp(event, window):
    """
    Handles cut/copy/paste operations

    Parameters:
    - event: The event to parse
    - window: Handle to the window we are interacting with

    """
    if event == "Copy":
        try:
            text = window["-INPUT-"].Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
        except tk.TclError:
            log("Nothing selected to copy!")
    elif event == "Copy ":
        try:
            text = window["-OUTPUT-"].Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
        except tk.TclError:
            log("Nothing selected to copy!")
    elif event == "Cut":
        try:
            text = window["-INPUT-"].Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
            window["-INPUT-"].Widget.delete("sel.first", "sel.last")
        except tk.TclError:
            log("Nothing selected to cut!")
    elif event == "Paste":
        try:
            # Check if there"s any text selected for replacement
            window["-INPUT-"].Widget.selection_get()
            # If there is, delete the selected text first
            window["-INPUT-"].Widget.delete("sel.first", "sel.last")
        except tk.TclError:
            # No text is selected; proceed without deleting
            pass  # No action required if no text is selected
        try:
            text = window.TKroot.clipboard_get()
            window["-INPUT-"].Widget.insert(tk.INSERT, text)
        except tk.TclError:
            log("Clipboard error - clipboard empty or contains non text data")


def clear_conversation(window):
    """
    Simple helper function to clear the chat log

    Parameters:
    - window: a handle to the window
    """
    ps = ProgramSettings()
    if ps.mode == "chat":
        window["-OUTPUT-"].update("")
    elif ps.mode == "story":
        window['-SYSTEM-'].update("")
        window['-CHARACTERS-'].update("")
        window['-STYLE-'].update("")
        window['-SUMMARY-'].update("")
        window['-PLOT-'].update("")
        window['-LENGTH-'].update("")
        window['-STARTER-'].update("")
        window['-Log-'].update("")
        window['-OUTPUT-'].update("")


def update_main_window(window):
    """
    Loads the previous conversational history into the chat log

    Parameters:
    - window: The window handle
    """
    ai = AI()
    ps = ProgramSettings()
    if ps.mode == "chat":
        clear_conversation(window)
        for i, message in enumerate(ai.core_memory):
            sender_role = message["role"]
            sender_name = ai.personality_definition["name"] if sender_role == "assistant" else ps.username
            sender_color = "purple" if sender_role == "assistant" else "blue"
            if ai.core_memory[i]["rating"] == "+":
                display_message(window, sender_name, message["content"], sender_color, "green")
            elif ai.core_memory[i]["rating"] == "-":
                display_message(window, sender_name, message["content"], sender_color, "red")
            else:
                display_message(window, sender_name, message["content"], sender_color, "black")
    elif ps.mode == "story":
        window['-SYSTEM-'].update(ai.system)
        window['-CHARACTERS-'].update(ai.characters)
        window['-STYLE-'].update(ai.style)
        window['-SUMMARY-'].update(ai.summary)
        window['-PLOT-'].update(ai.plot)
        window['-LENGTH-'].update(ai.length)
        window['-STARTER-'].update(ai.starter)


def handle_edit_event():
    """
    Opens a new window for editing the personality configuration. Updates the personality_definition
    dictionary based on user inputs from the edit window.
    """
    ai = AI()
    ps = ProgramSettings()
    numeric_fields = ("top_k", "top_p", "temperature", "response_length", "typical_p",  # A list of all the fields that should contain only numbers
                      "stm_size", "ltm_size", "length_penalty", "num_beams", "num_keywords", "repetition_penalty")
    edit_window = create_edit_window()
    valid_values = ai.personality_definition
    while True:
        event, values = edit_window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            edit_window.close()
            break
        if event == "Save":
            if ps.mode == "chat":
                for key, value in ai.personality_definition.items():
                    expected_type = type(value)
                    ai.personality_definition[key] = expected_type(values[key])
                log("Personality_defition updated.")
                # Split the text area content into lines, each representing a message"s content
                edited_contents = values["messages_editor"].split("\n")
                # Update system_messages with the new contents
                system_messages = [{"role": "system", "content": content} for content in edited_contents if content.strip()]
                ai.system_memory = system_messages
                update_hard_memory()
                edit_window.close()
                break
            if ps.mode == "story":
                for index, (key, value) in enumerate(ai.personality_definition.items()):
                    if index == 0:
                        continue  # Skip the rest of the loop for the first item
                    expected_type = type(value)
                    ai.personality_definition[key] = expected_type(values[key])

                update_hard_memory()
                edit_window.close()
        if event in numeric_fields:
            # We received a change in one of our number only fields
            if values[event]:  # Check if values[event] is not blank
                if not is_number(values[event]):  # Test if the user tried to stash a letter in our numbers
                    edit_window[event].update(valid_values[event])  # If so, revert to the last valid value
                else:
                    ttu = type(ai.personality_definition[event])
                    valid_values[event] = ttu(values[event])


def handle_create_event():
    """
    Opens a new window for editing the personality configuration. Updates the personality_definition
    dictionary based on user inputs from the edit window.
    """
    ai = AI()
    ps = ProgramSettings()
    numeric_fields = ("top_k", "top_p", "temperature", "response_length", "typical_p",  # A list of all the fields that should contain only numbers
                      "stm_size", "ltm_size", "length_penalty", "num_beams", "num_keywords", "repetition_penalty")
    window = create_edit_window()
    valid_values = ai.personality_definition
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            window.close()
            break
        if event == "Save":
            selected_folder = select_folder()
            if selected_folder != "":
                current_dir = selected_folder
                path = os.path.join(current_dir, values["name"])
                os.makedirs(path, exist_ok=False)
                ai.personality_path = path
                for key, value in ai.personality_definition.items():
                    expected_type = type(value)
                    ai.personality_definition[key] = expected_type(values[key])
                log("Personality_defition updated.")
                # Split the text area content into lines, each representing a message"s content
                edited_contents = values["messages_editor"].split("\n")
                # Update system_messages with the new contents
                system_messages = [{"role": "system", "content": content} for content in edited_contents if content.strip()]
                ai.system_memory = system_messages
                ps.personality_status = "loaded"
                update_hard_memory()
                window.close()
                break
        if event in numeric_fields:
            # We received a change in one of our number only fields
            if values[event]:  # Check if values[event] is not blank
                if not is_number(values[event]):  # Test if the user tried to stash a letter in our numbers
                    window[event].update(valid_values[event])  # If so, revert to the last valid value
                else:
                    ttu = type(ai.personality_definition[event])
                    valid_values[event] = ttu(values[event])


def popup_message(message):
    """
    Displays a notice to the user in the form of a popup window with an OK button

    Parameters:
    - message: The notice to display

    """
    layout = [
        [sg.Text(message)],
        [sg.Column([[sg.Button("OK")]], justification="center")]
    ]
    window = sg.Window("Notice!", layout, icon="./resources/bai.ico", modal=True)
    while True:
        event, _ = window.read()
        if event in (sg.WIN_CLOSED, "OK"):
            break
    window.close()


def select_folder():
    """
    Allows the user to select a folder and returns an empty string if no folder is selected.

    Parameters:
    - None

    Returns:
    - a string containing the folder path
    """
    folder = sg.popup_get_folder("Please select a folder:", no_window=True)
    if folder is None or len(folder) < 1:  # No folder was selected
        folder = ""
        log("No folder was selected!")
    else:
        log(f"Model selected: {folder}")
    return folder


def display_message(window, sender_name=None, message="", sender_color="black", message_color="black"):
    """
    Displays a message in the chat window with the specified formatting, including background color.

    Parameters:
    - window: The PySimpleGUI window object.
    - sender_name: The name of the message sender, string.
    - message: The message content, string.
    - sender_color: The color for the sender"s name, string.
    - message_color: The color for the message text, string.    - background_color: The background color for both sender"s name and message, string.
    """
    ps = ProgramSettings()
    if ps.mode == "chat":
        window["-OUTPUT-"].print(f"{sender_name}: ", text_color=sender_color, end="")
    window["-OUTPUT-"].print(message, text_color=message_color, end="\n")


def update_hard_memory(silent=0):
    """Update the AI"s "hard memory" - the hard drive copy of it"s memory"""
    ps = ProgramSettings()
    ai = AI()
    if ps.personality_status != "unloaded":
        if ps.mode == "chat":
            if ai.personality_definition["persistent"]:  # We save the conversation history only if persistence is enabled
                filtered_messages = [message for message in ai.core_memory if message["role"] != "system"]
            else:
                filtered_messages = []
            filtered_messages.insert(0, ai.personality_definition)
            personality_name = ai.personality_definition["name"]
            core_path = os.path.join(ai.personality_path, f"{personality_name}.json")
            sys_path = os.path.join(ai.personality_path, f"{personality_name}_system_messages.json")
            with open(core_path, "w", encoding="utf-8") as file:
                json.dump(filtered_messages, file, indent=4)
            with open(sys_path, "w", encoding="utf-8") as file:
                json.dump(ai.system_memory, file, indent=4)
        elif ps.mode == "story":
            filtered_messages = []
            filtered_messages.insert(0, ai.personality_definition)
            personality_name = ai.personality_definition["name"]
            current_status = {"system": ai.system, "characters": ai.characters, "plot": ai.plot, "style": ai.style, "summary": ai.summary, "length": ai.length, "starter": ai.starter}
            filtered_messages.insert(1, current_status)
            with open("./resources/Blissful Writer.json", 'w', encoding='utf-8') as file:
                json.dump(filtered_messages, file, indent=4)
    else:
        if silent == 0:
            log("Nothing to save!")


def handle_about_event():
    """
    Draws the about box and waits for ok/close
    """
    cpu_name = get_cpu_name()
    cpu_index = cpu_name.find("CPU")
    cpu_name = cpu_name[:cpu_index].strip() if cpu_index != -1 else cpu_name
    gpu_names = get_gpu_info()
    _, ram_total = get_ram_usage()
    os_name, os_version = get_os_name_and_version()
    ps = ProgramSettings()

    layout = [
        [sg.Column([[sg.Image("./resources/baiabout.png", size=(256, 256))]], justification="center", pad=((0, 36), (0, 0)))],
        [sg.Text("Blissful AI", justification="center", expand_x=True)],
        [sg.Text(f"Version {ps.VERSION}", justification="center", expand_x=True)],
        [sg.Text("BlissfulAI copyleft 2024 Blyss Sarania under CC-BY-NC-SA", justification="center", expand_x=True)],
        [sg.Text(f"CPU: {cpu_name}", justification="center", expand_x=True)],
        [sg.Text(f"RAM: {ram_total}GB", justification="center", expand_x=True)],
    ]

    # Adding GPU details. For multiple GPUs, create a text element for each
    for index, gpu in enumerate(gpu_names):
        layout.append([sg.Text(f"GPU {index}: {gpu}", justification="center", expand_x=True)])
    layout.append([sg.Text(f"OS: {os_name} {os_version}", justification="center", expand_x=True)])
    layout.append([sg.Column([[sg.Button("OK")]], justification="center")])
    # Window
    window = sg.Window("About BlissfulAI", layout, modal=True, finalize=True, icon="./resources/bai.ico")

    # Centering text (kind of a workaround since PySimpleGUI does not directly support centering multi-line text)
    for element in window.element_list():
        if isinstance(element, (sg.Button, sg.Text)):
            element.Widget.pack(expand=True)

    # Event Loop
    while True:
        event, _ = window.read()
        if event in [sg.WIN_CLOSED, "OK"]:
            break
        if event == "-LINK-":
            # Assuming open_url function is defined elsewhere or use webbrowser.open
            open_url("https://creativecommons.org/licenses/by-nc-sa/4.0/")

    window.close()


def create_edit_window():
    """
    Creates the window for editing the personality configuration

    Parameters:
    - None

    Returns:
    - window: a handle to the created window
    """
    def update_help(window, evname):
        # Explanations mapping
        explanations = {
            "name": "(name) - Specify the AI's name. This will be used in interactions and logging.",
            "top_p": "(top_p)(0.01-1.00)(0.85) - Controls the diversity of AI responses by limiting the next word choice to the top P percent. Helps in making the conversation more unpredictable.",
            "typical_p": "(typical_p)(0.01-1.00)(0.92) - Adjusts response diversity more smoothly compared to Top P by focusing on the most likely tokens, aiming for a balance between randomness and relevance.",
            "top_k": "(top_k)(1+)(50) - Limits the choices for the next word to the top K options to steer the conversation. A lower number can make responses more focused.",
            "temperature": "(temperature)(0.01+)(0.75) - Tweaks randomness in response generation. Higher values lead to more varied responses, while lower values make them more predictable.",
            "length_penalty": "(length_penalty)(0.00+)(1.00) - Penalizes longer responses to encourage more concise outputs. A higher penalty discourages rambling.",
            "repetition_penalty": "(repetition_penalty)(0.00+)(1.00) - Reduces repetition in responses. Higher values decrease the likelihood of repeating words, making responses more diverse.",
            "response_length": "(response_length)(1+)(64) - Sets the maximum length for responses. Longer lengths allow for more detailed answers but also take longer to create.",
            "stm_size": "(stm_size)(0+)(16) - Sets the number of short term memories to be selected for working memory. Incrases VRAM usage when increased.",
            "ltm_size": "(ltm_size)(0+)(20) - Sets the number of long term memories to be selected for working memory. Increases VRAM usage when increased.",
            "num_keywords": "(num_keywords)(1+)(3) - Number of keywords extracted from the input for context searching.",
            "num_beams": "(num_beams)(1+)(1) - Number of alternatives explored for generating responses. Higher numbers increase response quality at the cost of speed and VRAM.",
            "persistent": "(persistent) - Whether the AI's conversation history is persistent across sessions.",
            "messages_editor": "(system messages) - Defines the AI's personality"
        }

        window["explanation"].update(f"Help: {explanations[evname]}", text_color='green')
    ai = AI()
    ps = ProgramSettings()
    # Define the maximum label width for uniformity
    label_width = 20
    string_width = 16
    num_width = 16

    if ps.mode == "chat":
        # System messages section
        # Serialize system_messages for editing
        editable_messages = "\n".join([msg["content"] for msg in ai.system_memory])
        messages_editor = [[sg.Multiline(default_text=editable_messages, size=(120, 10), enable_events=True, key="messages_editor")]]
        # Update layout
        layout = [
            [sg.Text("Parameter:", size=(label_width, 1)), sg.Text("Value:", size=(14, 1)), sg.Text("Use?", size=(label_width, 1))],
            [sg.Text("Name", size=(label_width, 1)), sg.InputText(ai.personality_definition["name"], key="name", size=(string_width, 1), enable_events=True), sg.Text("")],
            [sg.Text("Top P", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_p"], key="top_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_p_enable"], key="top_p_enable")],
            [sg.Text("Typical P", size=(label_width, 1)), sg.InputText(ai.personality_definition["typical_p"], key="typical_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["typical_p_enable"], key="typical_p_enable")],
            [sg.Text("Top K", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_k"], key="top_k", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_k_enable"], key="top_k_enable")],
            [sg.Text("Temperature", size=(label_width, 1)), sg.InputText(ai.personality_definition["temperature"], key="temperature", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["temperature_enable"], key="temperature_enable")],
            [sg.Text("Length Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["length_penalty"], key="length_penalty", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["length_penalty_enable"], key="length_penalty_enable")],
            [sg.Text("Repetition Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["repetition_penalty"], key="repetition_penalty", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["repetition_penalty_enable"], key="repetition_penalty_enable")],
            [sg.Text("Response Length", size=(label_width, 1)), sg.InputText(ai.personality_definition["response_length"], key="response_length", size=(num_width, 1), enable_events=True)],
            [sg.Text("STM Size", size=(label_width, 1)), sg.InputText(ai.personality_definition["stm_size"], key="stm_size", size=(num_width, 1), enable_events=True)],
            [sg.Text("LTM Size", size=(label_width, 1)), sg.InputText(ai.personality_definition["ltm_size"], key="ltm_size", size=(num_width, 1), enable_events=True)],
            [sg.Text("Num keywords", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_keywords"], key="num_keywords", size=(num_width, 1), enable_events=True)],
            [sg.Text("Num Beams", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_beams"], key="num_beams", size=(num_width, 1), enable_events=True)],
            [sg.Text("Persistent", size=(label_width, 1)), sg.Checkbox("", default=ai.personality_definition["persistent"], key="persistent")],
            [sg.Text("Help: (parameter)(range)(starter value)", size=(80, 1), key="expl", text_color="green")],
            [sg.Text("Help: Explanations of parameters will appear here.", size=(80, 3), key="explanation", text_color="green")],
            [sg.Text("System Messages:", font=("Helvetica", 12, "underline"))],
            [sg.Column(messages_editor, vertical_alignment="top")],
            [sg.Button("Save"), sg.Button("Cancel")]
        ]
        event_names = [
            "name", "top_p", "typical_p", "top_k", "temperature",
            "length_penalty", "repetition_penalty", "response_length",
            "stm_size", "ltm_size", "num_keywords", "num_beams",
            "persistent", "messages_editor"
        ]

    elif ps.mode == "story":
        layout = [
            [sg.Text("Parameter:", size=(label_width, 1)), sg.Text("Value:", size=(14, 1)), sg.Text("Use?", size=(label_width, 1))],
            [sg.Text("Name", size=(label_width, 1)), sg.Text(ai.personality_definition["name"], size=(string_width - 2, 1), key="name", relief=sg.RELIEF_SUNKEN, background_color="#a3a3a3"), sg.Text("")],
            [sg.Text("Top P", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_p"], key="top_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_p_enable"], key="top_p_enable")],
            [sg.Text("Typical P", size=(label_width, 1)), sg.InputText(ai.personality_definition["typical_p"], key="typical_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["typical_p_enable"], key="typical_p_enable")],
            [sg.Text("Top K", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_k"], key="top_k", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_k_enable"], key="top_k_enable")],
            [sg.Text("Temperature", size=(label_width, 1)), sg.InputText(ai.personality_definition["temperature"], key="temperature", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["temperature_enable"], key="temperature_enable")],
            [sg.Text("Length Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["length_penalty"], key="length_penalty", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["length_penalty_enable"], key="length_penalty_enable")],
            [sg.Text("Repetition Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["repetition_penalty"], key="repetition_penalty", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["repetition_penalty_enable"], key="repetition_penalty_enable")],
            [sg.Text("Response Length", size=(label_width, 1)), sg.InputText(ai.personality_definition["response_length"], key="response_length", size=(num_width, 1), enable_events=True)],
            [sg.Text("Num Beams", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_beams"], key="num_beams", size=(num_width, 1), enable_events=True)],
            [sg.Text("Help: (parameter)(range)(starter value)", size=(80, 1), key="expl", text_color="green")],
            [sg.Text("Help: Explanations of parameters will appear here.", size=(80, 3), key="explanation", text_color="green")],
            [sg.Button("Save"), sg.Button("Cancel")]
        ]

        event_names = [
            "name", "top_p", "typical_p", "top_k", "temperature",
            "length_penalty", "repetition_penalty", "response_length",
            "num_beams"
        ]

    window = sg.Window("Edit Personality Configuration", layout, modal=True, finalize=True, icon="./resources/bai.ico")

    for evname in event_names:
        window[evname].Widget.bind('<Button-1>', lambda event, window=window, evname=evname: update_help(window, evname))
    return window


def create_settings_window():
    """
    Function for creating the window to edit the programs settings
    """
    def update_help(window, evname):
        explanations = {
            "username": "(Username) - What you wanna be called. Used in the chat log and certain templates.",
            "backend": "(Backend) - The backend to run inferences on. 'auto' is generally better than 'cuda' for CUDA.",
            "quant": "(Model Quantization) - The quantization to load the model with. Saves VRAM at a slight cost to accuracy.",
            "template": "(Model Template) - The template for formatting the model's input.",
            "default_model_path": "(Default Model) - The model to load on startup.",
            "default_personality_path": "(Default Personality) - The personality to load on startup.",
            "stream": "(Stream output to STDOUT?) - Whether to stream the generated tokens to stdout as they are generated.",
            "autosave": "(Autosave personality?) - If enabled, automatically save the loaded personality once a minute.",
            "auto_template": "(TRY to auto-select best): tries to automatically select the best template based on the model. If it fails, falls back to user selection."
        }
        window["explanation"].update(f"Help: {explanations[evname]}", text_color='green')

    ps = ProgramSettings()
    label_width = 20
    string_width = 36
    backend_options = ["cuda", "cpu", "auto"] if nvidia() is True else ["cpu", "auto"]
    quant_options = ["BNB 4bit", "BNB 4bit+", "BNB 8bit", "None"]
    template_options = ["HF Automatic", "BAI Zephyr", "BAI Opus", "BAI Alpaca", "BAI Instruct", "BAI SynthIA"]
    layout = [
        [sg.Text("Username:", size=(label_width, 1)), sg.Input(default_text=ps.username, key="username", enable_events=True)],
        [sg.Text("Backend:", size=(label_width, 1)), sg.Combo(backend_options, default_value=ps.backend, key="backend", readonly=True, enable_events=True)],
        [sg.Text("Model Quantization:", size=(label_width, 1)), sg.Combo(quant_options, default_value=ps.quant, key="quant", readonly=True, enable_events=True)],
        [sg.Text("Model Template:", size=(label_width, 1)), sg.Combo(template_options, default_value=ps.template, key="template", readonly=True, enable_events=True), sg.Checkbox("TRY to auto-select best", default=ps.auto_template, key="auto_template", enable_events=True)],
        [sg.Text("Default Model:", size=(label_width, 1)), sg.Input(default_text=ps.default_model, enable_events=True, key="default_model_path", size=(string_width, 1)), sg.FolderBrowse("Browse", target="default_model_path")],
        [sg.Text("Default Personality:", size=(label_width, 1)), sg.Input(default_text=ps.default_personality, enable_events=True, key="default_personality_path", size=(string_width, 1)), sg.FolderBrowse("Browse", target="default_personality_path")],
        [sg.Text("Stream output to STDOUT?", size=(label_width, 1)), sg.Checkbox("", default=ps.do_stream, key="stream", enable_events=True), sg.Text("Autosave personality?", size=(label_width, 1)), sg.Checkbox("", default=ps.autosave, key="autosave", enable_events=True)],
        [sg.Text("Help: Explanations of settings will appear here.", size=(60, 3), key="explanation", text_color="green")],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]
    window = sg.Window("Settings", layout, modal=True, icon="./resources/bai.ico", finalize=True)
    event_names = [
        "username", "backend", "quant", "template", "default_model_path",
        "default_personality_path", "stream", "auto_template", "autosave"
    ]

    for evname in event_names:
        window[evname].Widget.bind('<Button-1>', lambda _, window=window, evname=evname: update_help(window, evname))

    return window


def handle_settings_event():
    """
    Function for handling the settings event
    """
    ps = ProgramSettings()
    settings_window = create_settings_window()
    while True:
        event, values = settings_window.read(timeout=50)
        if event in (sg.WIN_CLOSED, "Cancel"):
            settings_window.close()
            break
        if event == "Save":
            if ps.model_status != "unloaded":
                ps.model_status = "reload needed" if ps.quant != values["quant"] else ps.model_status
            ps.backend = values["backend"]
            ps.quant = values["quant"]
            ps.default_model = values["default_model_path"]
            ps.default_personality = values["default_personality_path"]
            ps.do_stream = values["stream"]
            ps.username = values["username"]
            ps.template = values["template"]
            ps.auto_template = values["auto_template"]
            ps.autosave = values["autosave"]
            log("Settings updated.")
            ps.save_to_file()
            settings_window.close()
            break


def edit_response(initial_string):
    """
    A function to edit a string and return the edited value using the UI, with support for long strings.

    Parameters:
    - initial_string: The string we are working with

    Returns:
    - the edited string, or the original if no change
    """

    layout = [
        [sg.Text("Edit response:"), sg.Multiline(initial_string, size=(100, 3), key='MLINE')],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]
    window = sg.Window("Edit Response", layout, icon="./resources/bai.ico", modal=True)
    while True:
        event, values = window.read(timeout=50)
        if event in [sg.WIN_CLOSED, "Cancel"]:
            window.close()
            return initial_string
        if event == "Save":
            edited_string = values['MLINE'].rstrip()
            window.close()
            return edited_string

    window.close()


def create_guidance_message():
    """
    A function to create temporary system messages to guide the conversation
    """
    def update_display():
        current_messages = ""
        for i, entry in enumerate(ai.guidance_messages):
            current_messages += f"{i+1}: {entry['content']}; Turns remaining: {entry['turns']}\n"
        guidance_window["CURMSG"].update(current_messages)

    def handle_right_click(tk_event, guidance_window):
        if len(ai.guidance_messages) > 0:
            index = output_widget.index(f"@{tk_event.x},{tk_event.y}")
            line_number = int(index.split(".")[0]) - 1

            def delete_entry():
                log(f"Deleting {ai.guidance_messages[line_number]}...")
                del ai.guidance_messages[line_number]
                update_display()

            try:
                context_menu = tk.Menu(output_widget, tearoff=0)
                context_menu.add_command(label="Delete", command=delete_entry)
                context_menu.tk_popup(tk_event.x_root, tk_event.y_root)
            finally:
                context_menu.grab_release()
    ai = AI()

    layout = [
        [sg.Text("Current guidance:", size=(22, 1)), sg.Multiline(size=(100, 10), enable_events=True, disabled=True, key="CURMSG")],
        [sg.Text("New Guidance Message:", size=(22, 1)), sg.Multiline("", size=(100, 3), key='MLINE')],
        [sg.Text("Number of turns:", size=(22, 1)), sg.InputText("", size=(10, 1), key='TURNS')],
        [sg.Text("Help: Guidance messages are temporary system messages you can use to guide the conversation. They persist only for the set number of turns then disappear from memory. They are not saved between sessions. A 'turn' consists of one user input plus one AI response.", size=(80, 3), text_color="green")],
        [sg.Button("Add"), sg.Button("Close")]
    ]
    guidance_window = sg.Window("Create Guidance Message", layout, icon="./resources/bai.ico", modal=True, finalize=True)
    update_display()
    output_widget = guidance_window["CURMSG"].Widget
    output_widget.bind("<Button-3>", lambda tk_event: handle_right_click(tk_event, guidance_window))
    guidance_window['MLINE'].set_focus()
    while True:
        event, values = guidance_window.read(timeout=50)
        if event in [sg.WIN_CLOSED, "Close"]:
            guidance_window.close()
            break
        if event == "Add":
            if values['MLINE'] and values['TURNS']:
                new_message = values['MLINE'].rstrip()
                try:
                    num_turns = int(values['TURNS'])
                    ai.guidance_messages.append({"role": "system", "content": new_message, "turns": num_turns})
                    update_display()
                    guidance_window["MLINE"].update("")
                    guidance_window["TURNS"].update("")
                except ValueError:
                    popup_message("Number of turns must be an integer!")
            else:
                popup_message("Please fill out both a message and the number of turns before adding!")


def handle_middle_click(event, window, context_menu, last_entry):
    """
    Handles the middle click context menu

    Parameters:
    - event: The event we received
    - window: The main chat window, PSG windowobject
    - context_menu: The context menu to use, PSG context menu
    """
    ai = AI()
    ps = ProgramSettings()
    try:
        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")
        line_number = int(index.split(".")[0]) - 1

        def update_rating_up():
            if 0 <= line_number < len(ai.core_memory):
                ai.core_memory[line_number]["rating"] = "+"
                log(f"Line {line_number}: Thumbs up")
                update_main_window(window)

        def update_rating_neutral():
            if 0 <= line_number < len(ai.core_memory):
                ai.core_memory[line_number]["rating"] = ""
                log(f"Line {line_number}: Neutral")
                update_main_window(window)

        def update_rating_down():
            if 0 <= line_number < len(ai.core_memory):
                ai.core_memory[line_number]["rating"] = "-"
                log(f"Line {line_number}: Thumbs down")
                update_main_window(window)

        def regenerate_response():
            ps.special = "regenerate"

        def edit():
            ps.special = f"edit.{line_number}"

        if last_entry:
            context_menu.entryconfig("Regenerate", command=regenerate_response)
        context_menu.entryconfig("Edit", command=edit)
        context_menu.entryconfig("👍", command=update_rating_up)
        context_menu.entryconfig("-", command=update_rating_neutral)
        context_menu.entryconfig("👎", command=update_rating_down)

        # Display the context menu
        context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        # Make sure the menu is torn down properly
        context_menu.grab_release()


def update_context_menu(event, window):
    """
    Updates the middle click context menu in real time before displaying it to the user
    This way we can also use it to display context sensitive infos

    Parameters:
    - event: The event handle
    - window: The handle to the main window
    """
    ai = AI()
    widget = event.widget
    index = widget.index(f"@{event.x},{event.y}")
    line_number = int(index.split(".")[0]) - 1  # Assuming line numbers start at 1, adjust for 0-based indexing
    output_widget = window["-OUTPUT-"].Widget
    context_menu = tk.Menu(output_widget, tearoff=0)
    if line_number == len(ai.core_memory) - 1:
        last_entry = True
        context_menu.add_command(label="Regenerate")
    else:
        last_entry = False
    context_menu.add_command(label="Edit")
    context_menu.add_command(label="👍")
    context_menu.add_command(label="-")
    context_menu.add_command(label="👎")
    context_menu.add_command(label="Timestamp: " + ai.core_memory[line_number]["date"])
    context_menu.add_command(label="Identity: " + ai.core_memory[line_number]["identity"])
    handle_middle_click(event, window, context_menu, last_entry)


def create_chat_window():
    """
    Creates the main window for interacting with the AI

    Returns:
    - window: A handle to the created window
    """
    width, height = 1000, 500
    c_right_click_menu = ["", ["Copy "]]
    ccp_right_click_menu = ["", ["Copy", "Cut", "Paste"]]

    layout = [
        [sg.Multiline(size=(60, 20), key="-OUTPUT-", right_click_menu=c_right_click_menu, expand_y=True, enable_events=True, autoscroll=False, disabled=True, expand_x=True)],
        [sg.Text("", size=(40, 1), key="-NOTICE-", text_color="purple", expand_x=True), sg.Button("Guidance"), sg.Button("Load Model"), sg.Button("Create Personality"), sg.Button("Load Personality"), sg.Button("Edit Personality"), sg.Button("Settings")],
        [sg.Multiline(key="-INPUT-", size=(40, 3), expand_x=True, expand_y=True, right_click_menu=ccp_right_click_menu), sg.Button("Send", bind_return_key=True)],
        [sg.Text("", size=(80, 1), key="-STATUS-", text_color="black", expand_x=True), sg.Button("About")],
    ]
    window = sg.Window("BlissfulAI", layout, resizable=True, finalize=True, size=(width, height), icon="./resources/bai.ico")
    window['-INPUT-'].set_focus()
    # Bind to the resize event
    window.TKroot.bind("<Configure>", lambda event: enforce_minimum_size(window, width, height))
    window["-OUTPUT-"].Widget.config(selectbackground="#777777")
    window["-INPUT-"].Widget.config(selectbackground="#777777")
    output_widget = window["-OUTPUT-"].Widget

    # Bind click event
    output_widget.bind("<Button-2>", lambda event: update_context_menu(event, window))
    return window


def mode_selection_interface():
    """
    Creates and displays a window for the user to select between the modes
    """
    layout = [
        [sg.Text("Please choose your mode:", justification='center', expand_x=True)],
        [sg.Column([[sg.Button("Chat")], [sg.Button("Story")]], element_justification='center', expand_x=True)]
    ]
    # Create the window with the layout
    window = sg.Window("BlissfulAI", layout, resizable=True, finalize=True, icon="./resources/bai.ico")

    ps = ProgramSettings()
    while True:
        event, _ = window.read(timeout=100)
        if event == sg.WINDOW_CLOSED:
            graceful_shutdown()
            break
        if event == "Chat":
            log("Chat mode selected")
            ps.mode = "chat"
            break
        if event == "Story":
            log("Story mode selected")
            ps.mode = "story"
            break
    window.close()


def create_story_window():
    """
    Creates the main window for interacting with the AI in story mode

    - window: A handle to the created window
    """
    label_width = 8
    # width, height = 800, 500
    c_right_click_menu = ["", ["Copy "]]
    # ccp_right_click_menu = ["", ["Copy", "Cut", "Paste"]]

    layout = [
        [sg.Text("", size=(40, 1), key="-NOTICE-", text_color="purple", expand_x=True)],
        [
            sg.Text("System:", size=(label_width, 1)), sg.Multiline(size=(40, 6), key="-SYSTEM-", autoscroll=True),
            sg.Text("Plot:", size=(label_width, 1)), sg.Multiline(size=(40, 6), key="-PLOT-", autoscroll=True),
            sg.Column(  # Column for the Length label and input field
                [
                    [sg.Text("Length:", size=(label_width, 1))],
                    [sg.InputText(size=(10, 1), key="-LENGTH-")]
                ],
                element_justification="right"  # Aligns the elements in the column to the right
            )
        ],
        [
            sg.Text("Style:", size=(label_width, 1)), sg.Multiline(size=(40, 6), key="-STYLE-", autoscroll=True),
            sg.Text("Characters:", size=(label_width, 1)), sg.Multiline(size=(40, 6), key="-CHARACTERS-", autoscroll=True)
        ],
        [
            sg.Text("Summary:", size=(label_width, 1)), sg.Multiline(size=(40, 6), key="-SUMMARY-", autoscroll=True),
            sg.Text("Starter:", size=(label_width, 1)), sg.Multiline(size=(40, 6), key="-STARTER-", autoscroll=True),
            sg.Stretch(),  # This will push the elements to the right
            sg.Column(
                [
                    [sg.Button("Load Model", size=(10, 2))],
                    [sg.Button("Write!", size=(10, 1))],
                    [sg.Button("Edit Writer", size=(10, 2))]
                ],
                element_justification="right"
            )
        ],
        [
            sg.Text("Log:", size=(label_width, 1)), sg.Multiline(size=(40, 20), key="-LOG-", right_click_menu=c_right_click_menu, enable_events=True, autoscroll=True, disabled=True),
            sg.Text("Output:", size=(label_width, 1)), sg.Multiline(size=(40, 20), key="-OUTPUT-", right_click_menu=c_right_click_menu, enable_events=True, autoscroll=False, disabled=True)
        ],
        [sg.Stretch()],  # This will take up all the space in between
        [sg.Text("", size=(80, 1), key="-STATUS-", text_color="Purple", expand_x=True)]
    ]
    window = sg.Window("BlissfulAI", layout, font=("DejaVu Sans Mono", 10), resizable=True, finalize=True, icon="./resources/bai.ico")
    return window


def enforce_minimum_size(window, min_width, min_height):
    """
    Enforce the minimum size of the window.

    Parameters:
    - window: The window to enforce upon
    - min_width: Don"t let the width be less than this
    - min_height: Don"t let the height be less than this
    """
    width, height = window.size
    if width < min_width or height < min_height:
        new_width = max(width, min_width)
        new_height = max(height, min_height)
        window.TKroot.geometry(f"{new_width}x{new_height}")


@timed_execution
def load_personality(personality_path):
    """
    Function to load a specified AI and its memory into the system,
    along with making a backup of the personality configuration.

    Parameters:
    - personality_path: The path to the personality folder, a string

    """
    ai = AI()
    ps = ProgramSettings()
    # Define the path for easier access and readability
    personality_basename = os.path.basename(personality_path)
    config_path = os.path.join(personality_path, f"{personality_basename}.json") if ps.mode == "chat" else personality_path
    backup_path = os.path.join(personality_path, f"{personality_basename}_backup.json") if ps.mode == "chat" else f"{personality_path}_backup.json"
    system_message_path = os.path.join(personality_path, f"{personality_basename}_system_messages.json")
    # Check if the personality configuration file exists
    if os.path.exists(config_path):
        # Make a backup of the personality configuration file
        shutil.copy(config_path, backup_path)
        if ps.mode == "chat":
            with open(config_path, "r", encoding="utf-8") as file:  # Load in the  main personality file
                data = json.load(file)
                personality_definition = data[0]  # The first entry in this list of dictionaries is the personality_definition
                memory = data[1:]  # The remaining entries constitute it"s memory
                # Log the loaded personality details
                log(f"Loading personality ({personality_definition['name']})...")
                for key, value in personality_definition.items():
                    log(f"{key}: {value}")
                log("---------------------------------------------------------")
                # Load the system messages from the appropriate location
                ai.system_memory = []
                with open(system_message_path, "r", encoding="utf-8") as sm_file:
                    ai.system_memory = json.load(sm_file)
                log(f"AI System Memory: {ai.system_memory}")
                log(f"{len(ai.system_memory)} system messages.")
                memory = [{**item, "content": item["content"].replace("\n", "")} for item in memory]  # clean the \n
            memory_failed = 0
            log("Memory consistency checks...")
            fix_it = None
            for i, entry in enumerate(memory):
                try:
                    assert "role" in entry
                except AssertionError:
                    log(f"Memory {i} missing 'role' field!")
                    memory_failed += 1
                try:
                    assert "content" in entry
                except AssertionError:
                    log(f"Memory {i} missing 'content' field!")
                    memory_failed += 1
                try:
                    assert "identity" in entry
                except AssertionError:
                    log(f"Memory {i} missing 'identity' field!")
                    memory_failed += 1
                try:
                    assert "date" in entry
                except AssertionError:
                    log(f"Memory {i} missing 'date' field!")
                    memory_failed += 1
                try:
                    if "content" in entry and "identity" in entry and "date" in entry:
                        assert entry["identity"] == generate_hash(str(entry["content"]) + str(entry["date"]))
                except AssertionError:
                    log("Fingerprint mismatch for memory " + str(i))
                    log("Expected: " + generate_hash(str(entry["content"]) + str(entry["date"])))
                    log("Actual: " + entry["identity"])
                    if fix_it is None:
                        fix_it = handle_memory_failed()
                    if fix_it:
                        log("Updating...")
                        memory[i]["identity"] = generate_hash(str(entry["content"]) + str(entry["date"]))
                        log("Fingerprint: " + memory[i]["identity"])
                    else:
                        log("Failed memory ignored!")
                    memory_failed += 1
                try:
                    assert "rating" in entry
                except AssertionError:
                    log(f"Memory {i} missing 'rating' field!")
                    memory_failed += 1
            log("Detected " + str(memory_failed) + " errors in memory!")
            if fix_it is not None:
                if not fix_it:
                    log("Warning: Memory fingerprints NOT fixed, memory is inconsistent.")
                else:
                    log("Memory fingerprints updated!")
            log("Memory checks complete.")
            ai.personality_definition = personality_definition
            ai.core_memory = memory
            ai.personality_path = personality_path
            ps.personality_status = "loaded"
        elif ps.mode == "story":
            with open(config_path, 'r', encoding='utf-8') as file:  # Load in the core memory
                data = json.load(file)
                personality_definition = data[0]  # The first entry in this list of dictionaries is the personality_definition
                if len(data) > 1:
                    status = data[1]  # The remaining entries constitute it's memory
                    ai.system = status["system"]
                    ai.plot = status["plot"]
                    ai.characters = status["characters"]
                    ai.style = status["style"]
                    ai.summary = status["summary"]
                    ai.length = status["length"]
                    ai.starter = status["starter"]
                    # Log the loaded personality details
                    log(f"Loading personality ({personality_definition['name']})...")
                    for key, value in personality_definition.items():
                        log(f"{key}: {value}")
                    log("---------------------------------------------------------")
                ai.personality_definition = personality_definition
                ai.personality_path = personality_path
                ps.personality_status = "loaded"
    else:
        log("Personality configuration not found.")
        ps.personality_status = "unloaded"


def main():
    """
    The main function for creating the interface, loading and inferencing with models, editing the personality, etc
    Execution begins here.
    """
    if args.suppress:
        warnings.filterwarnings("ignore")
    sg.theme("Purple")
    llm = LanguageModel()
    ps = ProgramSettings()
    ps.do_stream = args.stream
    ps.backend = ps.backend.lower()
    model_response = Queue()
    model_queue = Queue()
    ps.load_from_file()  # load the settings from settings.json
    # Clear the previous logfile if it exists
    if os.path.exists("./logfile.txt"):
        os.remove("./logfile.txt")
    mode_selection_interface()
    ai = AI()  # The AI function returns either the "Chatter" or "Writer" objects depending on mode
    if ps.mode == "chat":
        # Initialize chat window UI.
        window = create_chat_window()
        # Load personality configuration and initialize core memory.
        if args.personality is not None and os.path.exists(args.personality):
            load_personality(args.personality)
        elif ps.default_personality is not None and os.path.exists(ps.default_personality):
            load_personality(ps.default_personality)
    elif ps.mode == "story":
        window = create_story_window()
        load_personality("./resources/Blissful Writer.json")
    # Display the conversation history
    update_main_window(window)

    def shutdown_handler(signum=None, frame=None):
        log("OS level exit request accepted.")
        log(f"Signum: {signum}, Frame: {frame}")
        ps = ProgramSettings()
        if ps.model_status == "inferencing":
            log("Removing last user_message to keep memory consistent.")
            ai.core_memory.pop()
        log("Final save of personality...")
        update_hard_memory()
        if window:
            window.close()
        os._exit(0)

    # For Windows
    if sys.platform == "win32":
        def windows_shutdown_handler(event):
            if event in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT, win32con.CTRL_CLOSE_EVENT, win32con.CTRL_LOGOFF_EVENT, win32con.CTRL_SHUTDOWN_EVENT):
                shutdown_handler()
                return True  # Indicate that the handler handled the event
            return False  # Other signals are not handled
        win32api.SetConsoleCtrlHandler(windows_shutdown_handler, True)

    # For Linux and others
    else:
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    if args.model is not None and os.path.exists(args.model):
        llm.model_path = args.model
        ps.model_status = "loading"
        threading.Thread(target=load_model, args=(llm.model_path, model_queue), daemon=True).start()
    elif ps.default_model is not None and os.path.exists(ps.default_model):
        llm.model_path = ps.default_model
        ps.model_status = "loading"
        threading.Thread(target=load_model, args=(llm.model_path, model_queue), daemon=True).start()

    # Main program loop
    ticks = 0
    while True:
        event, values = window.read(timeout=50)
        # The first section checks Window and ps.model_status events
        if event == sg.WIN_CLOSED:
            graceful_shutdown()
        elif ps.model_status == "inferencing":
            window["-NOTICE-"].update(f"{ai.personality_definition['name']} is typing" + animate_ellipsis())
            window.refresh()
        elif ps.model_status == "loading":
            window["-NOTICE-"].update("Model is loading" + animate_ellipsis())
            window.refresh()
        elif ps.model_status == "reload needed":
            log("Unloading current model...")
            llm.model = None
            llm.tokenizer = None
            gc.collect()
            if nvidia():
                if ps.backend in ["auto", "cuda"]:
                    torch.cuda.empty_cache()
            ps.model_status = "loading"
            threading.Thread(target=load_model, args=(llm.model_path, model_queue), daemon=True).start()

        # Handle special events
        if ps.special == "regenerate":
            if ps.model_status == "ready" and ps.personality_status == "loaded":
                log("Regenerating last response...")
                ps.special = ""
                ai.core_memory.pop()
                user_message = ai.core_memory[-1]["content"]
                ai.core_memory.pop()
                update_main_window(window)
                display_message(window, ps.username, user_message, "blue", "black")
                ps.model_status = "inferencing"
                threading.Thread(target=threaded_model_response, args=(llm, user_message, model_response), daemon=True).start()
            else:
                log("Cannot regenerate response when model is busy!")
                ps.special = ""
        elif ps.special.startswith("edit"):
            _, line_number = ps.special.split(".")
            line_number = int(line_number)
            log(f"Editing core memory {line_number}...")
            message = edit_response(ai.core_memory[line_number]["content"])
            if message != ai.core_memory[line_number]["content"]:
                log("Updating fingerprint...")
                now = str(datetime.now())
                identity = generate_hash(str(message) + str(now))
                ai.core_memory[line_number]["content"] = message
                ai.core_memory[line_number]["identity"] = identity
                ai.core_memory[line_number]["date"] = now
            ps.special = ""
            update_main_window(window)

        # Update the values of the story writer
        if ps.mode == "story":
            ai.system = values['-SYSTEM-']
            ai.characters = values['-CHARACTERS-']
            ai.plot = values['-PLOT-']
            ai.style = values['-STYLE-']
            ai.summary = values['-SUMMARY-']
            ai.length = values['-LENGTH-']
            ai.starter = values['-STARTER-']

        # Retrieve the model or response if it's sitting in queue
        if not model_queue.empty():
            llm.model, llm.tokenizer, llm.streamer = model_queue.get()
            ps.model_status = "ready"
            window["-NOTICE-"].update("")  # Clear the loading notice

        if not model_response.empty():
            output = model_response.get()
            if ps.mode == "chat":
                display_message(window, ai.personality_definition["name"], output, "purple", "black")
            elif ps.mode == "story":
                display_message(window, None, f"{ai.starter}{output}", None, "black")
            window["-NOTICE-"].update("")  # Clear the typing message.
            ps.model_status = "ready"

            # This big section is where we handle the buttons in the UI
        if event == "Send":
            if ps.personality_status == "loaded":
                if ps.model_status == "ready":
                    user_message = values["-INPUT-"]
                    display_message(window, ps.username, user_message, "blue", "black")
                    window["-INPUT-"].update("")  # Clear the input field
                    log("User message: " + user_message)
                    ps.model_status = "inferencing"
                    threading.Thread(target=threaded_model_response, args=(llm, user_message, model_response), daemon=True).start()
                elif ps.model_status in ["inferencing", "loading"]:
                    # Show a popup if the model is busy, loading, or not yet loaded.
                    popup_message(f"The model is currently {ps.model_status}. Please wait before sending messages.")
                elif ps.model_status == "unloaded":
                    popup_message("Please load a model first!")
            else:
                popup_message("No personality to send a message to, silly!")
        elif event == "Guidance":
            create_guidance_message()
        # Check if the model is busy and if not, (hopefully) unload the previous model and then start a thread to load the new model
        elif event == "Load Model":
            if ps.model_status == "ready":
                log("Unloading current model...")
                llm.model = None
                llm.tokenizer = None
                gc.collect()
                if nvidia():
                    if ps.backend in ["auto", "cuda"]:
                        torch.cuda.empty_cache()
            elif ps.model_status in ["inferencing", "loading"]:
                # Show a popup if the model is busy or loading.
                popup_message("The model is currently busy or loading.")
            if ps.model_status in ["ready", "unloaded"]:
                selected_folder = select_folder()
                if selected_folder == "":
                    log("No model selected!")
                    llm.model_path = None
                    ps.model_status = "unloaded"
                else:
                    llm.model_path = selected_folder
                    ps.model_status = "loading"
                    threading.Thread(target=load_model, args=(selected_folder, model_queue), daemon=True).start()
                    llm.model_path = selected_folder
        elif event == "Create Personality":
            if ps.personality_status == "loaded":
                log("Saving current personality...")
                update_hard_memory()
                ai.reset()
            handle_create_event()
        elif event == "Load Personality":
            if ps.model_status != "inferencing":
                if ps.personality_status == "loaded":
                    log("Saving current personality...")
                    update_hard_memory()
                    ai.reset()
                selected_folder = select_folder()
                if selected_folder == "":
                    log("No personality selected!")
                else:
                    load_personality(selected_folder)
                    update_main_window(window)
            else:
                popup_message("Cannot reload personality while the model is inferencing!")
        elif event == "Settings":
            if ps.model_status in ["ready", "unloaded"]:
                handle_settings_event()
            else:
                popup_message("Can't change settings while the model is loading or busy!")
        # Allows editing of the personality_defitinion and system messages
        elif event in ["Edit Personality", "Edit Writer"]:
            if ps.model_status != "inferencing":
                if ps.personality_status == "loaded":
                    update_hard_memory()
                    handle_edit_event()
                    load_personality(ai.personality_path)
                    update_main_window(window)
                else:
                    popup_message("No personality loaded to edit!")
            else:
                popup_message("Cannot edit personality while model is inferencing!")
        elif event == "About":
            handle_about_event()
        elif event in ["Copy", "Copy ", "Cut", "Paste"]:
            handle_ccp(event, window)
        elif event == "Write!":
            if ps.model_status == "ready":
                ai.system = values['-SYSTEM-']
                ai.characters = values['-CHARACTERS-']
                ai.plot = values['-PLOT-']
                ai.style = values['-STYLE-']
                ai.summary = values['-SUMMARY-']
                ai.length = values['-LENGTH-']
                ai.starter = values['-STARTER-']
                threading.Thread(target=threaded_story_response, args=(llm, model_response), daemon=True).start()
            elif ps.model_status in ["inferencing", "loading"]:
                # Show a popup if the model is busy, loading, or not yet loaded.
                popup_message(f"The model is currently {ps.model_status}. Please wait before sending messages.")
            elif ps.model_status == "unloaded":
                popup_message("Please load a model first!")
        ticks += 1
        if ticks % 5 == 0:
            update_system_status(window, llm.model_path)
        if ticks == 750:
            if ps.autosave:
                log("Autosaving personality...")
                update_hard_memory(1)
            ticks = 0


parser = argparse.ArgumentParser(description="")
parser.add_argument("-m", "--model", type=str, default=None)
parser.add_argument("-p", "--personality", type=str, default=None)
parser.add_argument("--suppress", type=bool, default=True)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("--stream", action="store_true")
args = parser.parse_args()
if __name__ == "__main__":
    main()
