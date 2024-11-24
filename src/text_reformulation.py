import os
import openai
import time


class TextCorrector:
    def __init__(self, api_key=None, input_file='output.txt', output_file='output_corrected.txt', chunk_size=4000):
        """
        Initialize the TextCorrector class.
        
        :param api_key: OpenAI API key (default: None, will use the environment variable if not provided)
        :param input_file: Path to the input file
        :param output_file: Path to the output file
        :param chunk_size: Maximum size of text chunks for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is not set.")
        openai.api_key = self.api_key
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size

    def read_input_file(self):
        """Read the input file and return its content."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return f.read()

    def write_output_file(self, text):
        """Write the corrected text to the output file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(text)

    def split_into_chunks(self, text):
        """Split the text into chunks of the specified size."""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def correct_chunk(self, chunk):
        """Correct a single chunk of text using the OpenAI API."""
        messages = [
            {
                "role": "system",
                "content": "أنت مساعد ذكي يقوم بتصحيح النصوص لغويًا ونحويًا مع الحفاظ على بنيتها الأصلية."
            },
            {
                "role": "user",
                "content": f"أعمل على تصحيح النص لغويًا ونحويًا مع الحفاظ على بنيته الأصلية:\n\n{chunk}"
            }
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=3500,
                n=1,
                temperature=0.7,
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError:
            print("Rate limit error, sleeping for 60 seconds")
            time.sleep(60)
            # Retry once after waiting
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=3500,
                    n=1,
                    temperature=0.7,
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Failed after retry: {e}")
                return chunk  # Fallback to original chunk
        except Exception as e:
            print(f"Error: {e}")
            return chunk  # Fallback to original chunk

    def process_text_with_openai(self):
        """Read, correct, and write the text."""
        # Read input file
        text = self.read_input_file()

        # Split text into chunks
        chunks = self.split_into_chunks(text)

        # Process each chunk
        reformulated_chunks = []
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            corrected_chunk = self.correct_chunk(chunk)
            reformulated_chunks.append(corrected_chunk)
            time.sleep(1)  # Sleep to respect rate limits

        # Concatenate corrected chunks
        output_text = ''.join(reformulated_chunks)

        # Write to output file
        self.write_output_file(output_text)

        print(f"Done. Corrected text written to {self.output_file}")
