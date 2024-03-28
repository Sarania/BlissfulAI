# -*- coding: utf-8 -*-
"""
SingletonMetas live here.
Created on Sat Mar 23 00:39:32 2024

@author: Blyss Sarania
"""
import os
import json
from datetime import datetime

class SingletonMeta(type):
    """
    The SingletonMeta class is useful for creating objects that persist as a single instance across the whole program. Basically a global class.
    """
    _instances = {}
    def __call__(cls, *Parameters, **kwParameters):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*Parameters, **kwParameters)
        return cls._instances[cls]

def log(input_string): #this is duplicated here to avoid dependencies during initial setup
    """
    Logs the string to the logfile

    Parameters:
    - input_string: The string to log.

    """
    with open("./logfile.txt", "a", encoding="utf-8") as logfile:
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        logfile.write(time_str + ": " + str(input_string) + "\n")
        print(time_str + ": " + str(input_string) + "\n")

class ProgramSettings(metaclass=SingletonMeta):
    """
    A class to hold the program's settings, ensuring that only one instance of the settings exists
    throughout the application lifecycle (singleton pattern).
    """

    def __init__(self):
        """
        Initializes the ProgramSettings with default values.
        """
        self._backend = "auto" #The backend to use, saved to file
        self._quant = "None" #The quantization to use, saved to file
        self._default_model = "" #The model to load on startup, saved to file
        self._default_personality = "" #The personality to load on startup, saved to file
        self._do_stream = False #Whether or not to stream the generated text to stdout, saved to file
        self._model_status = "unloaded" #Current status of the model, not saved
        self._personality_status = "unloaded" #Current status of the personality, not saved
        self._username = "User" #Username, saved to file
        self._template = "BAI Opus" #Selected template, saved to file
        self._cuda_version = "None" #
        self._VERSION = "0.9.5 RC1" #Program version

    @property
    def backend(self):
        """Gets the current backend setting."""
        return self._backend

    @backend.setter
    def backend(self, value):
        """Sets the backend setting.

        Parameters:
            value: The new backend setting value.
        """
        self._backend = value

    @property
    def quant(self):
        """Gets the current quantization setting."""
        return self._quant

    @quant.setter
    def quant(self, value):
        """Sets the quantization setting.

        Parameters:
            value: The new quantization setting value.
        """
        self._quant = value

    @property
    def default_model(self):
        """Gets the path to the default model file."""
        return self._default_model

    @default_model.setter
    def default_model(self, value):
        """Sets the path to the default model file, raising an error if the path does not exist.

        Parameters:
            value: The new path to the default model file.
        """
        if not os.path.exists(value):
            log("Specified model does not exist!")
        self._default_model = value

    @property
    def default_personality(self):
        """Gets the path to the default personality file."""
        return self._default_personality

    @default_personality.setter
    def default_personality(self, value):
        """Sets the path to the default personality file, raising an error if the path does not exist.

        Parameters:
            value: The new path to the default personality file.
        """
        if not os.path.exists(value):
            log("Specified personality path does not exist!")
        self._default_personality = value

    @property
    def do_stream(self):
        """Gets the current streaming setting."""
        return self._do_stream

    @do_stream.setter
    def do_stream(self, value):
        """Sets the streaming setting, ensuring it is a boolean value.

        Parameters:
            value: The new streaming setting (True or False).
        """
        if not isinstance(value, bool):
            raise ValueError("do_stream must be a boolean")
        self._do_stream = value

    @property
    def model_status(self):
        """Gets the current model status."""
        return self._model_status

    @model_status.setter
    def model_status(self, new_status):
        """Sets the current model status.

        Parameters:
            new_status: The new status of the model.
        """
        self._model_status = new_status

    @property
    def personality_status(self):
        """Gets the current personality module status."""
        return self._personality_status

    @personality_status.setter
    def personality_status(self, new_status):
        """Sets the current personality module status.

        Parameters:
            new_status: The new status of the personality module.
        """
        self._personality_status = new_status

    @property
    def username(self):
        """Gets the current username."""
        return self._username

    @username.setter
    def username(self, new_name):
        """Sets the username.

        Parameters:
            new_name: The new username.
        """
        self._username = new_name

    @property
    def template(self):
        """Gets the current template setting."""
        return self._template

    @template.setter
    def template(self, value):
        """Sets the template setting, ensuring it is a string.

        Parameters:
            value: The new template setting.
        """
        if not isinstance(value, str) and value is not None:
            raise ValueError("template must be a string")
        self._template = value

    @property
    def cuda_version(self):
        """Returns the set cuda version"""
        return self._cuda_version

    @cuda_version.setter
    def cuda_version(self, new_version):
        """Sets a new version of CUDA"""
        self._cuda_version = new_version

    @property
    def VERSION(self):
        """Gets the current version of the program."""
        return self._VERSION

    def save_to_file(self):
        """
        Saves the current settings to a JSON file named 'settings.json'.
        """
        data = {
            "_backend": self._backend,
            "_quant": self._quant,
            "_default_model": self._default_model,
            "_default_personality": self._default_personality,
            "_do_stream": self._do_stream,
            "_username": self._username,
            "_template": self._template,
        }
        # Write the dictionary to a file as JSON
        with open("./settings.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_file(cls):
        """
        Loads the settings from a 'settings.json' file, creating it with default settings if it doesn't exist.

        This method attempts to read the program's settings from a JSON file. If the file does not exist,
        it creates a new file with default settings. This ensures that the program can start with a known
        set of configurations and modify them as needed.

        Returns:
            ProgramSettings: An instance of ProgramSettings populated with the settings loaded from the file,
            or the default settings if the file was not previously available.
        """
        # Default settings as a fallback
        default_settings = {
            "_backend": "auto",
            "_quant": "None",
            "_default_model": "",
            "_default_personality": "",
            "_do_stream": False,
            "_username": "User",
            "_template": "BAI Opus"
            # Add or update other default settings here as needed.
        }

        # Check if the settings file exists
        if not os.path.exists("./settings.json"):
            # If the file does not exist, create it with default settings
            with open("./settings.json", "w", encoding="utf-8") as file:
                json.dump(default_settings, file, indent=4)
            log("No existing settings found. Created default settings file.")

        # Load the settings from the file
        with open("./settings.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        # Create an instance and populate it with the loaded or default settings
        instance = cls()
        instance.backend = data.get("_backend", default_settings["_backend"])
        instance.quant = data.get("_quant", default_settings["_quant"])
        instance.default_model = data.get("_default_model", default_settings["_default_model"])
        instance.default_personality = data.get("_default_personality", default_settings["_default_personality"])
        instance.do_stream = data.get("_do_stream", default_settings["_do_stream"])
        instance.username = data.get("_username", default_settings["_username"])
        instance.template = data.get("_template", default_settings["_template"])
        return instance

class AI(metaclass=SingletonMeta):
    """
    A singleton class designed to encapsulate all data and configurations related to the currently loaded personality. 
    It holds various memory models and settings that define the AI's personality and operational parameters.
    """

    def __init__(self):
        """
        Initializes the AI instance by setting up its default state, including resetting all forms of memory and settings.
        """
        self.reset()

    @property
    def personality_definition(self):
        """
        dict: Represents the set of parameters that define the AI's personality. Includes settings for response generation such as 'top_p', 'temperature', etc.
        """
        return self._personality_definition

    @personality_definition.setter
    def personality_definition(self, value):
        """
        Sets the AI's personality definition. Validates that the input is a dictionary representing the AI's personality settings.
        
        Args:
            value (dict): A dictionary containing the AI's personality settings.
            
        Raises:
            ValueError: If 'value' is not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("personality_definition must be a dictionary")
        self._personality_definition = value

    @property
    def core_memory(self):
        """
        list: The AI's core memory, intended for long-term storage of critical information.
        """
        return self._core_memory

    @core_memory.setter
    def core_memory(self, value):
        """
        Sets the AI's core memory. Validates that the input is a list.
        
        Args:
            value (list): A list representing the AI's core memory items.
            
        Raises:
            ValueError: If 'value' is not a list.
        """
        if not isinstance(value, list):
            raise ValueError("core_memory must be a list")
        self._core_memory = value

    @property
    def working_memory(self):
        """
        list: The AI's working memory, used for temporary storage and manipulation of information necessary for current tasks.
        """
        return self._working_memory

    @working_memory.setter
    def working_memory(self, value):
        """
        Sets the AI's working memory. Validates that the input is a list.
        
        Args:
            value (list): A list representing the AI's working memory items.
            
        Raises:
            ValueError: If 'value' is not a list.
        """
        if not isinstance(value, list):
            raise ValueError("working_memory must be a list")
        self._working_memory = value

    @property
    def system_memory(self):
        """
        list: The AI's system memory which holds the system messages
        """
        return self._system_memory

    @system_memory.setter
    def system_memory(self, value):
        """
        Sets the AI's system memory. Validates that the input is a list.
        
        Args:
            value (list): A list representing the AI's system memory items.
            
        Raises:
            ValueError: If 'value' is not a list.
        """
        if not isinstance(value, list):
            raise ValueError("system_memory must be a list")
        self._system_memory = value

    @property
    def num_sys_msg(self):
        """
        int: The number of system messages that have been processed by the AI. It must be a non-negative integer.
        """
        return self._num_sys_msg

    @num_sys_msg.setter
    def num_sys_msg(self, value):
        """
        Sets the number of system messages processed by the AI. Validates that the input is a non-negative integer.
        
        Args:
            value (int): The number of system messages to set.
            
        Raises:
            ValueError: If 'value' is not a non-negative integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("num_sys_msg must be a non-negative integer")
        self._num_sys_msg = value

    @property
    def personality_path(self):
        """
        str: The file path to the AI's personality definition file.
        """
        return self._personality_path

    @personality_path.setter
    def personality_path(self, new_path):
        """
        Sets the file path to the AI's personality definition.
        
        Args:
            new_path (str): The file path to set for the AI's personality definition.
        """
        self._personality_path = new_path

    def reset(self):
        """
        Resets the AI's state to default, clearing all memories and settings except for the singleton nature of the instance.
        """
        self.personality_definition = {"name": "Name", "top_p": 1.0, "top_k": 50, "temperature": 1.0, "response_length": 64, "persistent": True, "stm_size": 24, "ltm_size": 24, "repetition_penalty": 1.0, "length_penalty": 1.0, "num_beams": 1, "num_keywords": 3, "top_p_enable": False, "top_k_enable": False, "typical_p": 0.92, "typical_p_enable": True, "temperature_enable": False, "length_penalty_enable": False, "repetition_penalty_enable": False}
        self.core_memory = []
        self.working_memory = []
        self.system_memory = []
        self.num_sys_msg = 0
        self.personality_path = ""


class LanguageModel(metaclass=SingletonMeta):
    """
    Represents a language model within a singleton pattern to ensure availability across all modules.
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
        