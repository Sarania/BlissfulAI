# BlissfulAI

Welcome to **BlissfulAI**, a work in progress chatbot/story writing front end for LLMs. Note the story writing part is not yet released into the repository, but soon!

## Features

- Character driven interactive chat driven by large language models + guided uncensored creative story writing in one purple python package!
- Create customized characters and then interact with them! Any character you like! They can be your friend, your mentor, your lover, your advisor, and so much more!
- Characters develop from their interactions with you!
- Characters use a working memory system: a context sensitive combination of recent conversation(short term memory) and past conversations(long term memory).
- Storywriting mode allows you to specify an "author personality", plot outline, character descriptions, writing style, summary of previous events and more!
- Write the fanfic of your dreams! Rewrite the ending to {insert copyrighted work here} - but don't distribute it! Write an epic three part series about your dog! Your imagination is the limit.
- Fully customizable, adaptable to different hardware configurations. Low VRAM? Limp by with a quantized 3 billion parameter model. Max VRAM? Run giant models that the rest of us can only dream about!
- Voices, personality traits, character avatars and more are planned for the future!



## Prerequisites:
- Python3 on Windows, probably Linux as well(I tried, haven't tested yet though!)
- git
- Necessary CUDA Toolkit to go with your desired CUDA version (11.8 or 12.1)

## Beginning notes:

Inferencing requires a LOT of power and VRAM. It is recommended you have a modern Nvidia GPU with at least 8GB of memory. Example VRAM usage by model and quantization, numbers approximate:

- 1.6 billion parameters, 4bit NF4: ~1GB, 16 bit float: ~3.2GB
- 3 billion parameters, 4bit NF4: ~2GB, 16bit float: ~6GB
- 7 billion parameter, 4bit NF4: ~5GB, 16bit float: ~15GB
- 13 billion parameters, 4bit NF4: ~7.6GB, 16 bit float: ~28GB

VRAM usage will also increase depending on your personality settings, especially the size of short and long term memory.

## Basic Setup:

To get started with **BlissfulAI**, the first thing you need to do is clone it and set up the environment.

Clone the repository and change to it's directory on your local machine by running the following commands in your terminal/command prompt:

```
git clone https://github.com/Sarania/BlissfulAI.git
cd BlissfulAI
```

### Creating and Activating a Virtual Environment(recommended but not specifically required):

To create a virtual environment, run:

```python3 -m venv venv```


Activate the virtual environment:

- On Windows:

```.\venv\Scripts\activate```

- On Unix/MacOS:

```. ./venv/bin/activate```

### Installing Dependencies:

First you need to install Torch with or without CUDA depending on your needs. Go to the [Pytorch website](https://pytorch.org/get-started/locally/) and select according to your needs. Then paste and execute that link in your terminal.

For instance for Pytorch stable with CUDA 12.1 on Windows:

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

Lastly, install the remaining dependencies:

```pip3 install -r requirements.txt```

Now you're gonna need a model to work with. For best results I recommend at least a 7B model. Some good suggestions:

- [Opus 1.2 7B](https://huggingface.co/dreamgen/opus-v1.2-7b) - Highly recommended! An uncensored, 7 billion parameter model for roleplaying and creative writing. Use with the "BAI Opus" template. This is THE model for storywriting. 
- [OpenZephyrChat 7B](https://huggingface.co/Fredithefish/OpenZephyrChat-v0.2) - A 7 billion parameter model mainly used for chatting, a merge between OpenChat and Zephyr. Use with the "BAI Zephyr" template.
- [Zephyr 7B β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) - A 7 billion paramter model trained with DPO for good instruction following. Use with the "BAI Zephyr" or "HF Automatic template. Great for question answering tasks.
- [StableLM Zephyr 3B](https://huggingface.co/stabilityai/stablelm-zephyr-3b) - A pretty decent 3 billion parameter instruction following model. Recommended if VRAM is a concern, otherwise use a 7B model. Use with "BAI Zephyr" or "HF Automatic" template.

You can find other models on [HuggingFace](https://huggingface.co/models?sort=trending), and a good comparison exists [here](https://www.reddit.com/r/LocalLLaMA/comments/17fhp9k/huge_llm_comparisontest_39_models_tested_7b70b/). Downloading them can be annoying, so I wrote a script to make it easy. For instance to download the Opus 1.2 7B model from above:

```python download_model.py https://huggingface.co/dreamgen/opus-v1.2-7b```

This will download the model into the "opus-v1.2-7b" subdirectory of BlissfulAI, from which you can load it into the program.

You're now ready to go! Make sure to always activate the venv before running the program or it's utilities!

```python BlissfulAI.py```

## Usage Notes:

### Beginners:

When you first run the program, make sure to check out the Settings dialog and set up your backend, quantization, and other settings. Quantization is necessary for loading large models, explaining it is beyond the scope of this project. Suffice it to say, quantization sacrifices a little bit of quality for a lot less memory usage. If you are running out of VRAM, try using quantization. Next, you will need to create a new personality to interact with. Give it a name and adjust the settings to your liking. If you're unsure, leave the settings at default except for "Name" and the system messages section. This system messages define the character. [Example of some system messages and their effect.](/resources/baiexample.png) You can have as many system messages as you like, but too many tends to dillute their effect. I try to keep it around 8 or less. If you wish to know further information about the various parameters you can adjust, the internet is a good place to start but experimentation is best!

## Tips:
- If you're not using quantization and you have CUDA, it's highly recommended to set the backend to "auto" to let torch handle memory management! This can allow you to sometimes load larger models than you have VRAM, for instance I can load an unquantized 7B model.
- Moden Nvidia drivers support falling back to system memory when VRAM is exhausted. This is useful but WILL slow things down significantly if more than ~5% of the model is in system memory. Generally "auto" handles this more gracefully than "cuda" which will load the entire model into CUDA's context, taking up the full amount of VRAM.
- If quantization is enabled, the backend setting is ignored. This is because bitsandbytes automatically manages that in those cases.
- The character chat logs and parameters are just formatted JSON files. You can edit them freely but editing the messages themselves, the timestamp or the fingerprint WILL cause the memory integrity check to fail. You can fix this by passing "--fpfix" on the command line.

### Personal Data Warning:

BlissfulAI is a work in progress and receives a lot of updates. Also, I am not perfect(far from it!). At the same time, it's quite possible to develop intense emotional attachments to the AI characters one interacts with. For this reason, if you have a character you like, it is **STRONGLY** recommended you make a backup of the characters folder from time to time so that they don't get taken out from a bug! Because of the way LLM's work, ultimately the interactions you have with the characters you create form their personality. So if you lose those chat logs, even if you recreate everything else it won't be the same!

### License:

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
