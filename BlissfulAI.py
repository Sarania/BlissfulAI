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
import os
import sys
import gc
import json
import argparse
import threading
import shutil
import warnings
import tkinter as tk
from queue import Queue
import webbrowser
from inference_engine import threaded_model_response, load_model
from utils import log, timed_execution, is_number, update_system_status, animate_ellipsis, generate_hash, get_cpu_name, get_gpu_info, get_ram_usage, get_os_name_and_version, nvidia
from singletons import AI, LanguageModel, ProgramSettings
import torch
import win32api
import win32con
import PySimpleGUI as sg


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
    window["-OUTPUT-"].update("")


def update_conversation_history(window):
    """
    Loads the previous conversational history into the chat log
    
    Parameters:
    - window: The window handle
    """
    ai = AI()
    ps = ProgramSettings()
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


def handle_edit_event():
    """
    Opens a new window for editing the personality configuration. Updates the personality_definition
    dictionary based on user inputs from the edit window.
    """
    ai = AI()
    numeric_fields = ("top_k", "top_p", "temperature", "response_length", "typical_p", #A list of all the fields that should contain only numbers
                      "stm_size", "ltm_size", "length_penalty", "num_beams", "num_keywords", "repetition_penalty")
    edit_window = create_edit_window()
    valid_values=ai.personality_definition
    while True:
        event, values = edit_window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            edit_window.close()
            break
        if event == "Save":
            for key, value in ai.personality_definition.items():
                expected_type=type(value)
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
    numeric_fields = ("top_k", "top_p", "temperature", "response_length", "typical_p", #A list of all the fields that should contain only numbers
                      "stm_size", "ltm_size", "length_penalty", "num_beams", "num_keywords", "repetition_penalty")
    window = create_create_window()
    valid_values=ai.personality_definition
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            window.close()
            break
        if event == "Save":
            selected_folder=""
            while selected_folder == "":
                selected_folder=select_folder()

            current_dir = selected_folder
            path = os.path.join(current_dir, values["name"])
            os.makedirs(path, exist_ok=False)
            ai.personality_path=path
            for key, value in ai.personality_definition.items():
                expected_type=type(value)
                ai.personality_definition[key] = expected_type(values[key])
            log("Personality_defition updated.")
            # Split the text area content into lines, each representing a message"s content
            edited_contents = values["messages_editor"].split("\n")
            # Update system_messages with the new contents
            system_messages = [{"role": "system", "content": content} for content in edited_contents if content.strip()]
            ai.system_memory = system_messages
            ps.personality_status="loaded"
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
    Allows the user to select a folder
    
    Parameters:
    - None
    
    Returns:
    - The selected folder
    
    """
    folder = sg.popup_get_folder("Please select a folder:", no_window=True)
    if folder:  # Check if a folder was selected
        log(f"Model selected: {folder}")
        # Here you can do something with the selected folder path
    else:
        log("No folder was selected!")
    return folder

def display_message(window, sender_name, message, sender_color, message_color):
    """
    Displays a message in the chat window with the specified formatting, including background color.
    
    Parameters:
    - window: The PySimpleGUI window object.
    - sender_name: The name of the message sender, string.
    - message: The message content, string.
    - sender_color: The color for the sender"s name, string.
    - message_color: The color for the message text, string.    - background_color: The background color for both sender"s name and message, string.
    """
    window["-OUTPUT-"].print(f"{sender_name}: ", text_color=sender_color, end="")
    window["-OUTPUT-"].print(message, text_color=message_color, end="\n")


def update_hard_memory():
    """Update the AI"s "hard memory" - the hard drive copy of it"s memory"""
    ps = ProgramSettings()
    ai = AI()
    if ps.personality_status != "unloaded":
        if ai.personality_definition["persistent"]: # We save the conversation history only if persistence is enabled
            filtered_messages = [message for message in ai.core_memory if message["role"] != "system"]
        else:
            filtered_messages = []
        filtered_messages.insert(0, ai.personality_definition)
        personality_name = ai.personality_definition["name"]
        core_path = os.path.join(ai.personality_path, f"{personality_name}.json")
        sys_path = os.path.join(ai.personality_path, f"{personality_name}_system_messages.json")
        log(core_path)
        log(sys_path)
        with open(core_path, "w", encoding="utf-8") as file:
            json.dump(filtered_messages, file, indent=4)
        with open(sys_path, "w", encoding="utf-8") as file:
            json.dump(ai.system_memory, file, indent=4)
    else:
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

    # Updated Layout with Image
    layout = [
        [sg.Column([[sg.Image("./resources/baiabout.png", size=(256, 256))]], justification='center', pad=((0,36), (0,0)))],
        [sg.Text("Blissful AI", justification="center", expand_x=True)],
        [sg.Text(f"Version {ps.VERSION}", justification="center", expand_x=True)],
        [sg.Text("BlissfulAI copyleft 2024 Blyss Sarania under", justification="center", pad=((0,0),(0,0))), sg.Button("CC-BY-NC-SA", button_color=("blue", sg.theme_background_color()), border_width=0, tooltip="Click to visit license page", key="-LINK-", pad=((0,0),(0,0)))],
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
        if isinstance(element, sg.Text) or isinstance(element, sg.Button):
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
    ai=AI()
    # Define the maximum label width for uniformity
    label_width = 20
    string_width = 16
    num_width = 16
    # System messages section
    # Serialize system_messages for editing
    editable_messages = "\n".join([msg["content"] for msg in ai.system_memory])
    messages_editor = [[sg.Multiline(default_text=editable_messages, size=(120, 10), key="messages_editor")]]

    # Update layout
    layout = [
        [sg.Text("Parameter:", size=(label_width, 1)), sg.Text("Value:", size=(14, 1)), sg.Text("Use?", size=(label_width, 1))],
        [sg.Text("Name", size=(label_width, 1)), sg.InputText(ai.personality_definition["name"], key="name", size=(string_width, 1), enable_events=True), sg.Text("")],
        [sg.Text("Top P", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_p"], key="top_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_p_enable"], key="top_p_enable")],
        [sg.Text("Typical P", size=(label_width, 1)), sg.InputText(ai.personality_definition["typical_p"], key="typical_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["typical_p_enable"], key="typical_p_enable")],
        [sg.Text("Top K", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_k"], key="top_k", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_k_enable"], key="top_k_enable")],
        [sg.Text("Temperature", size=(label_width, 1)), sg.InputText(ai.personality_definition["temperature"], key="temperature", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["temperature_enable"], key="temperature_enable")],
        [sg.Text("Length Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["length_penalty"], key="length_penalty", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["length_penalty_enable"], key="length_penalty_enable")],
        [sg.Text("Repetition Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["repetition_penalty"], key="repetition_penalty", size=(num_width, 1), enable_events=True),  sg.Checkbox("", default=ai.personality_definition["repetition_penalty_enable"], key="repetition_penalty_enable")],
        [sg.Text("Response Length", size=(label_width, 1)), sg.InputText(ai.personality_definition["response_length"], key="response_length", size=(num_width, 1), enable_events=True)],
        [sg.Text("STM Size", size=(label_width, 1)), sg.InputText(ai.personality_definition["stm_size"], key="stm_size", size=(num_width, 1), enable_events=True)],
        [sg.Text("LTM Size", size=(label_width, 1)), sg.InputText(ai.personality_definition["ltm_size"], key="ltm_size", size=(num_width, 1), enable_events=True)],
        [sg.Text("Num keywords", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_keywords"], key="num_keywords", size=(num_width, 1), enable_events=True)],
        [sg.Text("Num Beams", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_beams"], key="num_beams", size=(num_width, 1), enable_events=True)],
        [sg.Text("Persistent", size=(label_width, 1)), sg.Checkbox("", default=ai.personality_definition["persistent"], key="persistent")],
        [sg.Text("System Messages:", font=("Helvetica", 12, "underline"))],
        [sg.Column(messages_editor, vertical_alignment="top")],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]

    window = sg.Window("Edit Personality Configuration", layout, modal=True, icon="./resources/bai.ico")
    return window

def create_create_window():
    """
    Creates the window for editing the personality configuration
    
    Parameters:
    - None
    
    Returns:
    - window: a handle to the created window
    """
    ai=AI()
    # Define the maximum label width for uniformity
    label_width = 20
    string_width = 16
    num_width = 16
    #The layout allows editing of system messages if they are enabled
    # System messages section
    messages_editor = [[sg.Multiline(size=(120, 10), key="messages_editor")]]

    # Update layout
    layout = [
        [sg.Text("Parameter:", size=(label_width, 1)), sg.Text("Value:", size=(14, 1)), sg.Text("Use?", size=(label_width, 1))],
        [sg.Text("Name", size=(label_width, 1)), sg.InputText(ai.personality_definition["name"], key="name", size=(string_width, 1), enable_events=True), sg.Text("")],
        [sg.Text("Top P", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_p"], key="top_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_p_enable"], key="top_p_enable")],
        [sg.Text("Typical P", size=(label_width, 1)), sg.InputText(ai.personality_definition["typical_p"], key="typical_p", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["typical_p_enable"], key="typical_p_enable")],
        [sg.Text("Top K", size=(label_width, 1)), sg.InputText(ai.personality_definition["top_k"], key="top_k", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["top_k_enable"], key="top_k_enable")],
        [sg.Text("Temperature", size=(label_width, 1)), sg.InputText(ai.personality_definition["temperature"], key="temperature", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["temperature_enable"], key="temperature_enable")],
        [sg.Text("Length Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["length_penalty"], key="length_penalty", size=(num_width, 1), enable_events=True), sg.Checkbox("", default=ai.personality_definition["length_penalty_enable"], key="length_penalty_enable")],
        [sg.Text("Repetition Penalty", size=(label_width, 1)), sg.InputText(ai.personality_definition["repetition_penalty"], key="repetition_penalty", size=(num_width, 1), enable_events=True),  sg.Checkbox("", default=ai.personality_definition["repetition_penalty_enable"], key="repetition_penalty_enable")],
        [sg.Text("Response Length", size=(label_width, 1)), sg.InputText(ai.personality_definition["response_length"], key="response_length", size=(num_width, 1), enable_events=True)],
        [sg.Text("STM Size", size=(label_width, 1)), sg.InputText(ai.personality_definition["stm_size"], key="stm_size", size=(num_width, 1), enable_events=True)],
        [sg.Text("LTM Size", size=(label_width, 1)), sg.InputText(ai.personality_definition["ltm_size"], key="ltm_size", size=(num_width, 1), enable_events=True)],
        [sg.Text("Num keywords", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_keywords"], key="num_keywords", size=(num_width, 1), enable_events=True)],
        [sg.Text("Num Beams", size=(label_width, 1)), sg.InputText(ai.personality_definition["num_beams"], key="num_beams", size=(num_width, 1), enable_events=True)],
        [sg.Text("Persistent", size=(label_width, 1)), sg.Checkbox("", default=ai.personality_definition["persistent"], key="persistent")],
        [sg.Text("System Messages:", font=("Helvetica", 12, "underline"))],
        [sg.Column(messages_editor, vertical_alignment="top")],
        [sg.Button("Save"), sg.Button("Cancel")]
    ]

    window = sg.Window("Create Personality Configuration", layout, modal=True, icon="./resources/bai.ico")
    return window

def create_settings_window():
    """
    Function for creating the window to edit the programs settings
    """
    ps = ProgramSettings()
    label_width = 20
    string_width = 36
    backend_options = ["cuda", "cpu", "auto"] if nvidia() is True else ["cpu", "auto"]
    quant_options = ["BNB 4bit", "BNB 4bit+", "BNB 8bit", "None"]
    template_options = ["HF Automatic", "BAI Zephyr", "BAI Opus", "BAI Alpaca", "BAI Instruct"]
    layout = [
    [sg.Text("Username:", size=(label_width, 1)), sg.Input(default_text=ps.username, key="username")],
    [sg.Text("Backend:", size=(label_width, 1)), sg.Combo(backend_options, default_value=ps.backend, key="backend", readonly=True)],
    [sg.Text("Model Quantization:", size=(label_width,1)), sg.Combo(quant_options, default_value=ps.quant, key="quant", readonly=True)],
    [sg.Text("Model Template:", size=(label_width,1)), sg.Combo(template_options, default_value=ps.template, key="template", readonly=True)],
    [sg.Text("Default Model:", size=(label_width, 1)), sg.Input(default_text=ps.default_model, key="default_model_path", size=(string_width, 1)), sg.FolderBrowse("Browse", target="default_model_path")],
    [sg.Text("Default Personality:", size=(label_width,1)), sg.Input(default_text=ps.default_personality, key="default_personality_path", size=(string_width, 1)), sg.FolderBrowse("Browse", target="default_personality_path")],
    [sg.Text("Stream output to STDOUT?", size=(label_width,1)), sg.Checkbox("", default=ps.do_stream, key="stream")],
    [sg.Button("Save"), sg.Button("Cancel")]
    ]
    window = sg.Window("Settings", layout, modal=True, icon="./resources/bai.ico")
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
            ps.model_status = "reload needed"  if ps.quant != values["quant"] else ps.model_status
            ps.backend = values["backend"]
            ps.quant = values["quant"]
            ps.default_model = values["default_model_path"]
            ps.default_personality = values["default_personality_path"]
            ps.do_stream = values["stream"]
            ps.username = values["username"]
            ps.template = values["template"]
            log("Settings updated.")
            ps.save_to_file()
            settings_window.close()
            break

def handle_middle_click(event, window, context_menu):
    """
    Handles the middle click context menu
    
    Parameters:
    - event: The event we received
    - window: The main chat window, PSG windowobject
    - context_menu: The context menu to use, PSG context menu
    """
    ai = AI()
    try:
        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")
        line_number = int(index.split(".")[0]) - 1

        def update_rating_up():
            if 0 <= line_number < len(ai.core_memory):
                ai.core_memory[line_number]["rating"] = "+"
                log(f"Line {line_number}: Thumbs up")
                update_conversation_history(window)

        def update_rating_neutral():
            if 0 <= line_number < len(ai.core_memory):
                ai.core_memory[line_number]["rating"] = ""
                log(f"Line {line_number}: Neutral")
                update_conversation_history(window)

        def update_rating_down():
            if 0 <= line_number < len(ai.core_memory):
                ai.core_memory[line_number]["rating"] = "-"
                log(f"Line {line_number}: Thumbs down")
                update_conversation_history(window)

        # Update context menu actions with the current line number in closure
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
    context_menu.add_command(label="👍")
    context_menu.add_command(label="-")
    context_menu.add_command(label="👎")
    context_menu.add_command(label="Timestamp: " + ai.core_memory[line_number]["date"])
    context_menu.add_command(label="Identity: " + ai.core_memory[line_number]["identity"])
    handle_middle_click(event, window, context_menu)


def create_chat_window():
    """
    Creates the main window for interacting with the AI
    
    Returns:
    - window: A handle to the created window
        
    """
    sg.theme("Purple")
    width, height = 900, 500
    c_right_click_menu = ["", ["Copy "]]
    ccp_right_click_menu = ["", ["Copy", "Cut", "Paste"]]

    layout = [
        [sg.Multiline(size=(60, 20), key="-OUTPUT-", right_click_menu=c_right_click_menu, enable_events=True, autoscroll=False, disabled=True, expand_y=True, expand_x=True)],
        [sg.Text("", size=(40, 1), key="-NOTICE-", text_color="purple", expand_x=True), sg.Button("Load Model"), sg.Button("Create Personality"), sg.Button("Load Personality"), sg.Button("Edit Personality"), sg.Button("Settings")],
        [sg.Multiline(key="-INPUT-", size=(40, 3), expand_x=True, right_click_menu=ccp_right_click_menu), sg.Button("Send", bind_return_key=True)],
        [sg.Text("", size=(80, 1), key="-STATUS-", text_color="black",expand_x=True), sg.Button("About")]
    ]
    window = sg.Window("BlissfulAI", layout, resizable=True, finalize=True, size=(width, height), icon="./resources/bai.ico")
    window["-INPUT-"].Widget.bind("<FocusIn>", "_FOCUS_IN_")
    # Bind to the resize event
    window.TKroot.bind("<Configure>", lambda event: enforce_minimum_size(window, width, height))
    window["-OUTPUT-"].Widget.config(selectbackground="#777777")
    window["-INPUT-"].Widget.config(selectbackground="#777777")
    output_widget = window["-OUTPUT-"].Widget

    # Bind click event
    output_widget.bind("<Button-2>", lambda event: update_context_menu(event, window))


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
    config_path = os.path.join(personality_path, f"{personality_basename}.json")
    backup_path = os.path.join(personality_path, f"{personality_basename}_backup.json")
    system_message_path = os.path.join(personality_path, f"{personality_basename}_system_messages.json")
    # Check if the personality configuration file exists
    if os.path.exists(config_path):
        # Make a backup of the personality configuration file
        shutil.copy(config_path, backup_path)

        with open(config_path, "r", encoding="utf-8") as file: #Load in the core memory
            data = json.load(file)
            personality_definition = data[0] # The first entry in this list of dictionaries is the personality_definition
            memory = data[1:]  # The remaining entries constitute it"s memory

            # Log the loaded personality details
            log(f"Loading personality ({personality_definition['name']})...")
            for key, value in personality_definition.items():
                log(f"{key}: {value}")
            log("---------------------------------------------------------")
            # Load the system messages from the appropriate location
            ai.system_memory = []
            with open(system_message_path, "r", encoding="utf-8") as sm_file:
                ai.system_memory=json.load(sm_file)
            ai.num_sys_msg=len(ai.system_memory)
            log(ai.system_memory)
            log(str(ai.num_sys_msg) + " system messages.")
            memory = [{**item, "content": item["content"].replace("\n", "")} for item in memory] # clean the \n
        memory_failed=0
        log("Memory consistency checks...")
        for i, entry in enumerate(memory):
            try:
                assert "role" in entry
            except AssertionError:
                log(f"Memory {i} missing 'role' field!")
                memory_failed+=1
            try:
                assert "content" in entry
            except AssertionError:
                log(f"Memory {i} missing 'content' field!")
                memory_failed+=1
            try:
                assert "identity" in entry
            except AssertionError:
                log(f"Memory {i} missing 'identity' field!")
                memory_failed+=1
            try:
                assert "date" in entry
            except AssertionError:
                log(f"Memory {i} missing 'date' field!")
                memory_failed+=1
            try:
                if "content" in entry and "identity" in entry and "date" in entry:
                    assert entry["identity"] == generate_hash(str(entry["content"]) + str(entry["date"]))
            except AssertionError:
                log("Fingerprint mismatch for memory " + str(i))
                if args.fpfix:
                    log("Updating...")
                    memory[i]["identity"] = generate_hash(str(entry["content"]) + str(entry["date"]))
                    log("Fingerprint: " + memory[i]["identity"])
                else:
                    log("Expected: " + generate_hash(str(entry["content"]) + str(entry["date"])))
                    log("Actual: " + entry["identity"])
                    print("Pass --fpfix to attempt to correct this!")
                    memory_failed+=1
            try:
                assert "rating" in entry
            except AssertionError:
                log(f"Memory {i} missing 'rating' field!")
                memory_failed+=1
        log("Detected " + str(memory_failed) + " errors in memory!")
        assert memory_failed==0
        log("Memory checks passed.")
        ai.personality_definition=personality_definition
        ai.core_memory = memory
        ai.personality_path=personality_path
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
    ai = AI()
    llm = LanguageModel()
    ps = ProgramSettings()
    ps.do_stream=args.stream
    ps.backend=ps.backend.lower()
    model_response = Queue()
    model_queue = Queue()
    ps.load_from_file() #load the settings from settings.json
    # Clear the previous logfile if it exists
    if os.path.exists("./logfile.txt"):
        os.remove("./logfile.txt")

    # Load personality configuration and initialize core memory.
    if args.personality is not None:
        load_personality(args.personality)
    elif ps.default_personality is not None:
        load_personality(ps.default_personality)
    if "top_p_enable" not in ai.personality_definition:
        ai.personality_definition["top_p_enable"] = False
    if "top_k_enable" not in ai.personality_definition:
        ai.personality_definition["top_k_enable"] = False
    if "typical_p" not in ai.personality_definition:
        ai.personality_definition["typical_p"] = 1.0
    if "typical_p_enable" not in ai.personality_definition:
        ai.personality_definition["typical_p_enable"] = False
    if "temperature_enable" not in ai.personality_definition:
        ai.personality_definition["temperature_enable"] = False
    if "length_penalty_enable" not in ai.personality_definition:
        ai.personality_definition["length_penalty_enable"] = False
    if "repetition_penalty_enable" not in ai.personality_definition:
        ai.personality_definition["repetition_penalty_enable"] = False
    # Initialize chat window UI.
    window = create_chat_window()

    def shutdown_handler(event):
        if event in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT, win32con.CTRL_CLOSE_EVENT, win32con.CTRL_LOGOFF_EVENT, win32con.CTRL_SHUTDOWN_EVENT):
            log("OS level exit request accepted.")
            ps=ProgramSettings()
            if ps.model_status=="inferencing":
                log("Removing last user_message to keep memory consistent.")
                ai.core_memory.pop()
            log("Final save of personality...")
            update_hard_memory()
            if window:
                window.close()
            os._exit(0)
        return False  # Other signals are not handled

    # Display the conversation history
    update_conversation_history(window)

    #Set this up to try to catch shutdown/restart etc
    win32api.SetConsoleCtrlHandler(shutdown_handler, True)

    if args.model is not None:
        llm.model_path = args.model
        ps.model_status="loading"
        threading.Thread(target=load_model, args=(llm.model_path, model_queue), daemon=True).start()
    elif ps.default_model is not None:
        llm.model_path = ps.default_model
        ps.model_status="loading"
        threading.Thread(target=load_model, args=(llm.model_path, model_queue), daemon=True).start()


    # Main program loop
    while True:
        event, values = window.read(timeout=50)
        if event == sg.WIN_CLOSED:
            graceful_shutdown()
        elif ps.model_status == "inferencing":
            window["-NOTICE-"].update(f"{ai.personality_definition['name']} is typing" + animate_ellipsis())
            window.refresh()
        elif ps.model_status == "loading":
            window["-NOTICE-"].update("Model is loading" + animate_ellipsis())
            window.refresh()
        elif ps.model_status == "reload needed":
            ps.model_status="loading"
            threading.Thread(target=load_model, args=(llm.model_path, model_queue), daemon=True).start()

        if not model_queue.empty():
            llm.model, llm.tokenizer, llm.streamer = model_queue.get()
            ps.model_status = "ready"
            window["-NOTICE-"].update("")  # Clear the loading notice

        if not model_response.empty():
            output=model_response.get()
            display_message(window, ai.personality_definition["name"], output, "purple", "black")
            window["-NOTICE-"].update("")  # Clear the typing message.
            ps.model_status = "ready"

        # This section handles the response from the UI based on the user"s input
        if event == "Send":
            if ps.personality_status == "loaded":
                if ps.model_status == "ready":
                    user_message = values["-INPUT-"]
                    display_message(window, ps.username, user_message, "blue", "black")
                    window["-INPUT-"].update("")  # Clear the input field
                    log("User message: " + user_message)
                    ps.model_status="inferencing"
                    threading.Thread(target=threaded_model_response, args=(user_message, model_response), daemon=True).start()
                elif ps.model_status in ["inferencing", "loading"]:
                    # Show a popup if the model is busy, loading, or not yet loaded.
                    popup_message(f"The model is currently {ps.model_status}. Please wait before sending messages.")
                elif ps.model_status == "unloaded":
                    popup_message("Please load a model first!")
            else:
                popup_message("No personality to send a message to, silly!")
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
                selected_folder=select_folder()
                if selected_folder == "":
                    log("No model selected!")
                    llm.model_path=None
                    ps.model_status="unloaded"
                else:
                    llm.model_path = selected_folder
                    ps.model_status="loading"
                    threading.Thread(target=load_model, args=(selected_folder, model_queue), daemon=True).start()
                    llm.model_path = selected_folder
        elif event =="Create Personality":
            if ps.personality_status == "loaded":
                log("Saving current personality...")
                update_hard_memory()
                ai.reset()
            handle_create_event()
        elif event =="Load Personality":
            if ps.model_status != "inferencing":
                if ps.personality_status == "loaded":
                    log("Saving current personality...")
                    update_hard_memory()
                    ai.reset()
                selected_folder=select_folder()
                if selected_folder == "":
                    log("No personality selected!")
                else:
                    load_personality(selected_folder)
                    update_conversation_history(window)
            else:
                popup_message("Cannot reload personality while the model is inferencing!")
        elif event =="Settings":
            handle_settings_event()
        #Allows editing of the personality_defitinion and system messages
        elif event == "Edit Personality":
            if ps.model_status != "inferencing":
                if ps.personality_status == "loaded":
                    update_hard_memory()
                    handle_edit_event()
                    load_personality(ai.personality_path)
                    update_conversation_history(window)
                else:
                    popup_message("No personality loaded to edit!")
            else:
                popup_message("Cannot edit personality while model is inferencing!")
        elif event =="About":
            handle_about_event()
        elif event in ["Copy", "Copy ", "Cut", "Paste"]:
            handle_ccp(event, window)
        update_system_status(window, llm.model_path)


parser = argparse.ArgumentParser(description="")
parser.add_argument("-m", "--model", type=str, default=None)
parser.add_argument("-p", "--personality", type=str, default=None)
parser.add_argument("--suppress", type=bool, default=True)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("--fpfix", action="store_true")
parser.add_argument("--stream", action="store_true")
args = parser.parse_args()
if __name__ == "__main__":
    main()
