
# Whether Expert

this is the definitive compendium of doubts and certitudes about whether and how to use the openai assistant-beta v2 assistants api functions and tools including the well known function calling, file search with vector stores, code interpreter, and structured response with output schema.

## Features

- Fetches temperature data for specified cities with function calling.
- Retrieves humidity data from vector stores and uploaded files with file search.
- Uses the OpenAI Code Interpreter to calculate differences between cities' temperature and humidity.
- Builds the responts in compliance with a json schema.

## Requirements

- **Python 3.7+**
- **OpenAI API Key**: The app requires an OpenAI API key to access OpenAI's services.
- **OpenAI Python Library**: The app is tested with version **1.51** of the OpenAI Python library.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/whether-expert.git
    cd whether-expert
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:
    - Export your API key as an environment variable:
      ```bash
      export OPENAI_API_KEY='your_openai_api_key'
      ```
    - On Windows:
      ```cmd
      set OPENAI_API_KEY=your_openai_api_key
      ```

## Running the App

echo "## Running the App

Run the Python script:
\`\`\`bash
python whether.py
\`\`\`

### Possible Parameters:
- **\`--reset\`**: Resets and deletes all created resources (assistants, vector stores, etc.) when provided.
  \`\`\`bash
  python whether.py --reset
  \`\`\`

- **\`-m\` or \`--model\`**: Specify the model to be used for the assistant (e.g., \`gpt-4o\`, \`gpt-4o-mini\`).
  \`\`\`bash
  python whether.py --model gpt-4o
  \`\`\`

These parameters give you control over how the app runs, allowing you to reset resources or choose different OpenAI models.
" >> README.md



The assistant will process the requested cities and provide weather data such as temperature and humidity differences.

## Notes

- The app was tested with **OpenAI Python Library v1.51**. Ensure compatibility when running it on different versions.
- For any errors or issues, check if your OpenAI API key is correctly set up and ensure you're using the correct Python version.

## License

This project is licensed under the MIT License.
