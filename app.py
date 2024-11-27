import os
import subprocess
import json
from datetime import timedelta
from transformers import pipeline
import tempfile
import re
import gradio as gr
import groq
from groq import Groq


# setup groq 

client = Groq(api_key="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")

def handle_groq_error(e, model_name):
    error_data = e.args[0]

    if isinstance(error_data, str):
        # Use regex to extract the JSON part of the string
        json_match = re.search(r'(\{.*\})', error_data)
        if json_match:
            json_str = json_match.group(1)
            # Ensure the JSON string is well-formed
            json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
            error_data = json.loads(json_str)

    if isinstance(e, groq.AuthenticationError):
        if isinstance(error_data, dict) and 'error' in error_data and 'message' in error_data['error']:
            error_message = error_data['error']['message']
            raise gr.Error(error_message)
    elif isinstance(e, groq.RateLimitError):
        if isinstance(error_data, dict) and 'error' in error_data and 'message' in error_data['error']:
            error_message = error_data['error']['message']
            error_message = re.sub(r'org_[a-zA-Z0-9]+', 'org_(censored)', error_message) # censor org
            raise gr.Error(error_message)
    else:
        raise gr.Error(f"Error during Groq API call: {e}")


# language codes for subtitle maker

LANGUAGE_CODES = {
    "English": "en",
    "Chinese": "zh",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Polish": "pl",
    "Catalan": "ca",
    "Dutch": "nl",
    "Arabic": "ar",
    "Swedish": "sv",
    "Italian": "it",
    "Indonesian": "id",
    "Hindi": "hi",
    "Finnish": "fi",
    "Vietnamese": "vi",
    "Hebrew": "he",
    "Ukrainian": "uk",
    "Greek": "el",
    "Malay": "ms",
    "Czech": "cs",
    "Romanian": "ro",
    "Danish": "da",
    "Hungarian": "hu",
    "Tamil": "ta",
    "Norwegian": "no",
    "Thai": "th",
    "Urdu": "ur",
    "Croatian": "hr",
    "Bulgarian": "bg",
    "Lithuanian": "lt",
    "Latin": "la",
    "Māori": "mi",
    "Malayalam": "ml",
    "Welsh": "cy",
    "Slovak": "sk",
    "Telugu": "te",
    "Persian": "fa",
    "Latvian": "lv",
    "Bengali": "bn",
    "Serbian": "sr",
    "Azerbaijani": "az",
    "Slovenian": "sl",
    "Kannada": "kn",
    "Estonian": "et",
    "Macedonian": "mk",
    "Breton": "br",
    "Basque": "eu",
    "Icelandic": "is",
    "Armenian": "hy",
    "Nepali": "ne",
    "Mongolian": "mn",
    "Bosnian": "bs",
    "Kazakh": "kk",
    "Albanian": "sq",
    "Swahili": "sw",
    "Galician": "gl",
    "Marathi": "mr",
    "Panjabi": "pa",
    "Sinhala": "si",
    "Khmer": "km",
    "Shona": "sn",
    "Yoruba": "yo",
    "Somali": "so",
    "Afrikaans": "af",
    "Occitan": "oc",
    "Georgian": "ka",
    "Belarusian": "be",
    "Tajik": "tg",
    "Sindhi": "sd",
    "Gujarati": "gu",
    "Amharic": "am",
    "Yiddish": "yi",
    "Lao": "lo",
    "Uzbek": "uz",
    "Faroese": "fo",
    "Haitian": "ht",
    "Pashto": "ps",
    "Turkmen": "tk",
    "Norwegian Nynorsk": "nn",
    "Maltese": "mt",
    "Sanskrit": "sa",
    "Luxembourgish": "lb",
    "Burmese": "my",
    "Tibetan": "bo",
    "Tagalog": "tl",
    "Malagasy": "mg",
    "Assamese": "as",
    "Tatar": "tt",
    "Hawaiian": "haw",
    "Lingala": "ln",
    "Hausa": "ha",
    "Bashkir": "ba",
    "jw": "jw",
    "Sundanese": "su",
}





# helper functions

def split_audio(input_file_path, chunk_size_mb):
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    file_number = 1
    chunks = []
    with open(input_file_path, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            chunk_name = f"{os.path.splitext(input_file_path)[0]}_part{file_number:03}.mp3" # Pad file number for correct ordering
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunks.append(chunk_name)
            file_number += 1
            chunk = f.read(chunk_size)
    return chunks

def merge_audio(chunks, output_file_path):
    with open("temp_list.txt", "w") as f:
        for file in chunks:
            f.write(f"file '{file}'\n")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe", "0",
                "-i",
                "temp_list.txt",
                "-c",
                "copy",
                "-y",
                output_file_path
            ],
            check=True
        )
        os.remove("temp_list.txt")
        for chunk in chunks:
            os.remove(chunk)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Error during audio merging: {e}")


# Checks file extension, size, and downsamples or splits if needed.

ALLOWED_FILE_EXTENSIONS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
MAX_FILE_SIZE_MB = 25
CHUNK_SIZE_MB = 25

def check_file(input_file_path):
    if not input_file_path:
        raise gr.Error("Please upload an audio/video file.")

    file_size_mb = os.path.getsize(input_file_path) / (1024 * 1024)
    file_extension = input_file_path.split(".")[-1].lower()

    if file_extension not in ALLOWED_FILE_EXTENSIONS:
        raise gr.Error(f"Invalid file type (.{file_extension}). Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}")

    if file_size_mb > MAX_FILE_SIZE_MB:
        gr.Warning(
            f"File size too large ({file_size_mb:.2f} MB). Attempting to downsample to 16kHz MP3 128kbps. Maximum size allowed: {MAX_FILE_SIZE_MB} MB"
        )

        output_file_path = os.path.splitext(input_file_path)[0] + "_downsampled.mp3"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_file_path,
                    "-ar",
                    "16000",
                    "-ab",
                    "128k",
                    "-ac",
                    "1",
                    "-f",
                    "mp3",
                    "-y",
                    output_file_path,
                ],
                check=True
            )

            # Check size after downsampling
            downsampled_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
            if downsampled_size_mb > MAX_FILE_SIZE_MB:
                gr.Warning(f"File still too large after downsampling ({downsampled_size_mb:.2f} MB). Splitting into {CHUNK_SIZE_MB} MB chunks.")
                return split_audio(output_file_path, CHUNK_SIZE_MB), "split"

            return output_file_path, None
        except subprocess.CalledProcessError as e:
            raise gr.Error(f"Error during downsampling: {e}")
    return input_file_path, None


# subtitle maker

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def json_to_srt(transcription_json):
    srt_lines = []

    for segment in transcription_json:
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        text = segment['text']

        srt_line = f"{segment['id']+1}\n{start_time} --> {end_time}\n{text}\n"
        srt_lines.append(srt_line)

    return '\n'.join(srt_lines)

def extract_audio(input_file):
    audio_file = "extracted_audio.wav"
    command = f"ffmpeg -i {input_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_file}"
    subprocess.run(command, shell=True, check=True)
    return audio_file

def transcribe_audio(input_file):
    # Extract audio from video file if needed
    if input_file.lower().endswith(('.mp4', '.mkv', '.avi', '.webm')):
        input_file = extract_audio(input_file)

    # Check if the file exists and has valid audio
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Audio file not found: {input_file}")
    
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=-1)

    try:
        transcription = asr_pipeline(input_file)
        return transcription['text']
    except Exception as e:
        raise ValueError(f"Error during transcription: {e}")

# Example usage


def generate_subtitles(input_mode, input_file, link_input, prompt, language, auto_detect_language, model, font_selection, font_file, font_color, font_size, outline_thickness, outline_color):
    if model == "local using whisper-large-v3":
        return generate_subtitles_local(input_mode, input_file, link_input, prompt, language, auto_detect_language, model, include_video, font_selection, font_file, font_color, font_size, outline_thickness, outline_color)
    if input_mode == "Upload Video/Audio File":
        input_file_path = input_file
    
    include_video = True
    processed_path, split_status = check_file(input_file_path)
    full_srt_content = ""
    total_duration = 0
    segment_id_offset = 0

    if split_status == "split":
        srt_chunks = []
        video_chunks = []
        for i, chunk_path in enumerate(processed_path):
            try:
                with open(chunk_path, "rb") as file:
                    transcription_json_response = client.audio.transcriptions.create(
                        file=(os.path.basename(chunk_path), file.read()),
                        model=model,
                        prompt=prompt,
                        response_format="verbose_json",
                        language=None if auto_detect_language else language,
                        temperature=0.0,
                    )
                transcription_json = transcription_json_response.segments

                # Adjust timestamps and segment IDs
                for segment in transcription_json:
                    segment['start'] += total_duration
                    segment['end'] += total_duration
                    segment['id'] += segment_id_offset
                segment_id_offset += len(transcription_json)
                total_duration += transcription_json[-1]['end']  # Update total duration

                srt_content = json_to_srt(transcription_json)
                full_srt_content += srt_content
                temp_srt_path = f"{os.path.splitext(chunk_path)[0]}.srt"
                with open(temp_srt_path, "w", encoding="utf-8") as temp_srt_file:
                    temp_srt_file.write(srt_content)
                    temp_srt_file.write("\n") # add a new line at the end of the srt chunk file to fix format when merged
                srt_chunks.append(temp_srt_path)

                if include_video and input_file_path.lower().endswith((".mp4", ".webm")):
                    try:
                        output_file_path = chunk_path.replace(os.path.splitext(chunk_path)[1], "_with_subs" + os.path.splitext(chunk_path)[1])
                        # Handle font selection
                        if font_selection == "Custom Font File" and font_file:
                            font_name = os.path.splitext(os.path.basename(font_file.name))[0]  # Get font filename without extension
                            font_dir = os.path.dirname(font_file.name)  # Get font directory path
                        elif font_selection == "Custom Font File" and not font_file:
                            font_name = None  # Let FFmpeg use its default Arial
                            font_dir = None  # No font directory
                            gr.Warning(f"You want to use a Custom Font File, but uploaded none. Using the default Arial font.")
                        elif font_selection == "Arial":
                            font_name = None  # Let FFmpeg use its default Arial
                            font_dir = None  # No font directory
                            
                        # FFmpeg command
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                chunk_path,
                                "-vf",
                                f"subtitles={temp_srt_path}:fontsdir={font_dir}:force_style='Fontname={font_name},Fontsize={int(font_size)},PrimaryColour=&H{font_color[1:]}&,OutlineColour=&H{outline_color[1:]}&,BorderStyle={int(outline_thickness)},Outline=1'",
                                "-preset", "fast",
                                output_file_path,
                            ],
                            check=True,
                        )
                        video_chunks.append(output_file_path) 
                    except subprocess.CalledProcessError as e:
                        raise gr.Error(f"Error during subtitle addition: {e}")     
                elif include_video and not input_file_path.lower().endswith((".mp4", ".webm")):
                    gr.Warning(f"You have checked on the 'Include Video with Subtitles', but the input file {input_file_path} isn't a video (.mp4 or .webm). Returning only the SRT File.", duration=15)
            except groq.AuthenticationError as e:
                handle_groq_error(e, model)
            except groq.RateLimitError as e:
                handle_groq_error(e, model)
                gr.Warning(f"API limit reached during chunk {i+1}. Returning processed chunks only.")
                if srt_chunks and video_chunks:
                    merge_audio(video_chunks, 'merged_output_video.mp4')
                    with open('merged_output.srt', 'w', encoding="utf-8") as outfile:
                        for chunk_srt in srt_chunks:
                            with open(chunk_srt, 'r', encoding="utf-8") as infile:
                                outfile.write(infile.read())
                    return 'merged_output.srt', 'merged_output_video.mp4'
                else:
                    raise gr.Error("Subtitle generation failed due to API limits.")

        # Merge SRT chunks
        final_srt_path = os.path.splitext(input_file_path)[0] + "_final.srt"
        with open(final_srt_path, 'w', encoding="utf-8") as outfile:
            for chunk_srt in srt_chunks:
                with open(chunk_srt, 'r', encoding="utf-8") as infile:
                    outfile.write(infile.read())

        # Merge video chunks
        if video_chunks:
            merge_audio(video_chunks, 'merged_output_video.mp4')
            return final_srt_path, 'merged_output_video.mp4'
        else:
            return final_srt_path, None

    else:  # Single file processing (no splitting)
        try:
            with open(processed_path, "rb") as file:
                transcription_json_response = client.audio.transcriptions.create(
                    file=(os.path.basename(processed_path), file.read()),
                    model=model,
                    prompt=prompt,
                    response_format="verbose_json",
                    language=None if auto_detect_language else language,
                    temperature=0.0,
                )
            transcription_json = transcription_json_response.segments

            srt_content = json_to_srt(transcription_json)
            temp_srt_path = os.path.splitext(input_file_path)[0] + ".srt"
            with open(temp_srt_path, "w", encoding="utf-8") as temp_srt_file:
                temp_srt_file.write(srt_content)

            if include_video and input_file_path.lower().endswith((".mp4", ".webm")):
                try:
                    output_file_path = input_file_path.replace(
                        os.path.splitext(input_file_path)[1], "_with_subs" + os.path.splitext(input_file_path)[1]
                    )
                    # Handle font selection
                    if font_selection == "Custom Font File" and font_file:
                        font_name = os.path.splitext(os.path.basename(font_file.name))[0]  # Get font filename without extension
                        font_dir = os.path.dirname(font_file.name)  # Get font directory path
                    elif font_selection == "Custom Font File" and not font_file:
                        font_name = None  # Let FFmpeg use its default Arial
                        font_dir = None  # No font directory
                        gr.Warning(f"You want to use a Custom Font File, but uploaded none. Using the default Arial font.")
                    elif font_selection == "Arial":
                        font_name = None  # Let FFmpeg use its default Arial
                        font_dir = None  # No font directory

                    # FFmpeg command
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            input_file_path,
                            "-vf",
                            f"subtitles={temp_srt_path}:fontsdir={font_dir}:force_style='FontName={font_name},Fontsize={int(font_size)},PrimaryColour=&H{font_color[1:]}&,OutlineColour=&H{outline_color[1:]}&,BorderStyle={int(outline_thickness)},Outline=1'",
                            "-preset", "fast",
                            output_file_path,
                        ],
                        check=True,
                    )
                    return temp_srt_path, output_file_path
                except subprocess.CalledProcessError as e:
                    raise gr.Error(f"Error during subtitle addition: {e}")
            elif include_video and not input_file_path.lower().endswith((".mp4", ".webm")):
                gr.Warning(f"You have checked on the 'Include Video with Subtitles', but the input file {input_file_path} isn't a video (.mp4 or .webm). Returning only the SRT File.", duration=15)
            
            return temp_srt_path, None
        except groq.AuthenticationError as e:
            handle_groq_error(e, model)
        except groq.RateLimitError as e:
            handle_groq_error(e, model)
        except ValueError as e:
            raise gr.Error(f"Error creating SRT file: {e}")






theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="neutral"
).set(
    border_color_primary='*neutral_300',
    block_border_width='1px',
    block_border_width_dark='1px',
    block_title_border_color='*secondary_100',
    block_title_border_color_dark='*secondary_200',
    input_background_fill_focus='*secondary_300',
    input_border_color='*border_color_primary',
    input_border_color_focus='*secondary_500',
    input_border_width='1px',
    input_border_width_dark='1px',
    slider_color='*secondary_500',
    slider_color_dark='*secondary_600'
)

css = """
.gradio-container{max-width: 1400px !important}
h1{text-align:center}
.extra-option {
    display: none;
}
.extra-option.visible {
    display: block;
}
"""



with gr.Blocks(theme=theme, css=css) as interface:
    gr.Markdown(
        """
    # Fast Subtitle Maker
    Inference by Groq API  
    If you are having API Rate Limit issues, you can retry later based on the [rate limits](https://console.groq.com/docs/rate-limits) or <a href="https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank"> <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a> with <a href=https://console.groq.com/keys>your own API Key</a> </p>
    Original Hugging Face Space by [Nick088] 
      
    """
    )

    with gr.Column():
        # Input mode selection
        input_mode = gr.Dropdown(choices=["Upload Video/Audio File"], value="Upload Video/Audio File", label="Input Mode")
        # Input components
        input_file = gr.File(label="Upload Audio/Video", file_types=[f".{ext}" for ext in ALLOWED_FILE_EXTENSIONS], visible=True)
        link_input_info = gr.Markdown("Using yt-dlp to download Youtube Video Links + other platform's ones. Check [all supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)!", visible=False)
        link_input = gr.Textbox(label="Enter Video/Audio Link", visible=False)

    # Model and options
    model_choice_subtitles = gr.Dropdown(choices=["whisper-large-v3", "distil-whisper-large-v3-en", "local using whisper-large-v3"], value="whisper-large-v3", label="Audio Speech Recogition (ASR) Model")
    transcribe_prompt_subtitles = gr.Textbox(label="Prompt (Optional)", info="Specify any context or spelling corrections.")
    with gr.Row():
        language_subtitles = gr.Dropdown(choices=[(lang, code) for lang, code in LANGUAGE_CODES.items()], value="en", label="Language")
        auto_detect_language_subtitles = gr.Checkbox(label="Auto Detect Language")
    
    with gr.Row(visible=True) as subtitle_video_settings:
        with gr.Column():
            font_selection = gr.Radio(["Arial", "Custom Font File"], value="Arial", label="Font Selection", info="Select what font to use")
            font_file = gr.File(label="Upload Font File (TTF or OTF)", file_types=[".ttf", ".otf"], visible=False)
        font_color = gr.ColorPicker(label="Font Color", value="#FFFFFF")
        font_size = gr.Slider(label="Font Size (in pixels)", minimum=10, maximum=60, value=24, step=1)
        outline_thickness = gr.Slider(label="Outline Thickness", minimum=0, maximum=5, value=1, step=1)
        outline_color = gr.ColorPicker(label="Outline Color", value="#000000")

    # Generate button
    transcribe_button_subtitles = gr.Button("Generate Subtitles")

    # Output and settings
    
    gr.Markdown("The SubText Rip (SRT) File, contains the subtitles, you can upload this to any video editing app for adding the subs to your video and also modify/stilyze them")
    srt_output = gr.File(label="SRT Output File")
    

    
    video_output = gr.Video(label="Output Video with Subtitles", visible=True)


    # Event bindings

    # input mode
    def toggle_input(mode):
        if mode == "Upload Video/Audio File":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    input_mode.change(fn=toggle_input, inputs=[input_mode], outputs=[input_file, link_input_info, link_input])

    
    # show custom font file selection
    font_selection.change(lambda font_selection: gr.update(visible=font_selection == "Custom Font File"), inputs=[font_selection], outputs=[font_file])
    
    # Update language dropdown based on model selection
    def update_language_options(model):
        if model == "distil-whisper-large-v3-en":
            return gr.update(choices=[("English", "en")], value="en", interactive=False)
        else:
            return gr.update(choices=[(lang, code) for lang, code in LANGUAGE_CODES.items()], value="en", interactive=True)

    model_choice_subtitles.change(fn=update_language_options, inputs=[model_choice_subtitles], outputs=[language_subtitles])

    # Modified generate subtitles event
    transcribe_button_subtitles.click(
        fn=generate_subtitles,
        inputs=[
            input_mode,
            input_file,
            link_input,
            transcribe_prompt_subtitles,
            language_subtitles,
            auto_detect_language_subtitles,
            model_choice_subtitles,
            font_selection,
            font_file,
            font_color,
            font_size,
            outline_thickness,
            outline_color,
        ],
        outputs=[srt_output, video_output],
    )

interface.launch(share=True)