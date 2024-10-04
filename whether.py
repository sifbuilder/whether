#  whether.py

# Standard library imports
import argparse
import io
import inspect
import json
import os
import sys
import time
import traceback
import uuid
from pprint import pprint
from typing import Any, Callable, Dict, List, Tuple, Union

# Third-party imports
import aiohttp
import jsonschema
import openai
from openai import OpenAI

#   ===========================================================

#   Abstract

#   ===========================================================
#   1. Models

def sist_get_model(model_arg=None):
    MODEL_MAP = {
        "o": "gpt-4o",
        "o-min1": "gpt-4o-mini",
        "o1": "gpt-4o1-preview",
        "o1-min1": "gpt-4o1-mini"
    }
    
    # Default model
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # If model_arg is None or an empty string, return the default model
    if model_arg is None or model_arg.strip() == "":
        print(f"No model specified. Defaulting to {DEFAULT_MODEL}.")
        return DEFAULT_MODEL
    
    # If the argument exists in the dictionary, return the corresponding model
    if model_arg in MODEL_MAP:
        return MODEL_MAP[model_arg]
    else:
        # If the argument is not in the dictionary, return the default model
        print(f"Unknown model argument '{model_arg}'. Defaulting to {DEFAULT_MODEL}.")
        return DEFAULT_MODEL

def sist_get_api():
    api = {"OpenAI-Beta": "assistants=v2"}
    return api

def sist_get_client():
    """
    This function initializes the OpenAI client.
    Note: The app was tested with version 1.51.
    """
    try:
        # Expected version for comparison
        tested_version = "1.51"
        
        # Fetch the OpenAI API key from the environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("The OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
        
        # Set the API key
        openai.api_key = openai_api_key
        
        # Print the actual version of the OpenAI library
        actual_version = openai.__version__
        print(f"Tested with OpenAI version: {tested_version}")
        print(f"Running with OpenAI version: {actual_version}")
        
        # Compare the versions
        if actual_version != tested_version:
            print(f"Warning: The current OpenAI version ({actual_version}) is different from the tested version ({tested_version}).")
        else:
            print("The current OpenAI version matches the tested version.")
        
        # Initialize the OpenAI client
        client = OpenAI(default_headers=sist_get_api())
        return client
    
    except ValueError as e:
        # Handle the missing API key error and display it nicely
        print(f"\nError occurred in {__file__}, function 'sist_get_client' at line {inspect.currentframe().f_lineno}:")
        print(f"{type(e).__name__}: {e}\n")
        raise



#   ===========================================================
#   2. Asssitant

def sist_get_instructions(goal, tasks=[], functions=[], use_vector_stores=False, use_code_interpreter=False, output_schema=None):
    """
    Generate assistant instructions based on the goal, functions (list of functions), and tasks.
    """
    instructions = f"""
    You are an assistant whose goal is to {goal}.

    You have the following tools to help you:
    """

    tool_index = 1

    # Loop through the functions (az_ent_specs) to generate instructions
    for i, tool in enumerate(functions, 1):
        tool_name = tool['function']['name']
        tool_description = tool['function']['description']
        instructions += f"{tool_index}. Use the **{tool_name}** function to {tool_description}.\n"

        tool_index += 1

    if use_vector_stores:
        instructions += f"{tool_index}. If a relevant function is not availble, locate the relevant data in the files uploaded to the **vector stores**.\n"

        tool_index += 1

    if use_code_interpreter:
        instructions += f"{tool_index}. Use the **Code Interpreter** tool for arithmetic calculations and data analysis.\n"

        tool_index += 1

    #  output schema
    if output_schema:
        instructions += f"{tool_index}. Your response MUST strictly comply with the following JSON schema: {json.dumps(output_schema, indent=2)}.\n"
        tool_index += 1
        instructions += f"{tool_index}. Do not use Markdown formatting, code blocks, or any other text wrapping.\n"
        tool_index += 1
        instructions += f"{tool_index}. The response should be a valid JSON object that can be directly parsed by a JSON parser.\n"
        tool_index += 1
        instructions += f"{tool_index}. Do not include any explanations or additional text outside of the JSON object.\n"


    instructions += "\n### Tasks:\n"
    # Add the tasks
    for task in tasks:
        instructions += f"- {task}\n"

    instructions += """
    Your response must include all relevant data and their differences for each requested item.

    Make sure each step of the process is completed in the correct order and that all results are gathered before formulating your response.
    """
    return instructions


def sist_create_assistant(client, instructions, model, tools=None, vector_store_id=None):
    """
    Wrapper function to create an assistant using the client.beta.assistants.create method.

    Parameters:
    - client: The client object to interact with the API.
    - instructions: The instructions to give the assistant.
    - model: The model to use for the assistant (e.g., "gpt-4o").
    - tools: A list of tools to enable for the assistant (optional).
    - vector_store_id: The vector store ID to be used by the file_search tool (optional).
    
    Returns:
    - assistant_id: The ID of the created assistant.
    """
    try:
        print("Creating assistant with the following configurations:")
        print(f"Instructions: {instructions}")
        print(f"Model: {model}")
        print(f"Tools: {tools if tools else 'No tools provided'}")
        print(f"Vector Store ID: {vector_store_id}\n")

        # Prepare tool resources dynamically based on vector_store_id
        tool_resources = {}
        if vector_store_id:
            tool_resources["file_search"] = {
                "vector_store_ids": [vector_store_id]
            }

        # Create the assistant with or without the file_search tool depending on vector_store_id
        assistant = client.beta.assistants.create(
            instructions=instructions,
            model=model,
            tools=tools if tools else None,  # Pass tools only if provided
            tool_resources=tool_resources if tool_resources else None  # Only include tool_resources if not empty
        )
        assistant_id = assistant.id
        return assistant_id

    except Exception as e:
        print(f"An error occurred while creating the assistant: {str(e)}")
        traceback.print_exc()
        raise


def sist_find_assistant(client, assistant_name_prefix=None):
    # If no assistant_name_prefix is provided, return None immediately
    if assistant_name_prefix is None:
        return None

    assistants = list(client.beta.assistants.list())
    
    for assistant in assistants:
        if assistant and hasattr(assistant, 'name') and assistant.name is not None and assistant.name.startswith(assistant_name_prefix):
            print(f"Found assistant with prefix {assistant_name_prefix} and id: {assistant.id}")
            return assistant.id
    
    print(f"Could not find assistant with prefix {assistant_name_prefix}")
    return None


#   ===========================================================
#   3. File and Vector Store Management

def sist_upload_data_file(client, data_file):
    """
    Upload the specified file to OpenAI and return the file ID.
    """
    with open(data_file, "rb") as file:
        file_contents = file.read().decode("utf-8")
        print(f"Contents of '{data_file}' before uploading:\n{file_contents}\n")
        file.seek(0)
        file_upload = client.files.create(file=file, purpose='assistants')

        # Check if the file already exists in the client's file list
        existing_files = list(client.files.list())
        for file in existing_files:
            print(f"file {file.filename} : {file.id}")

    return file_upload.id

import io

def sist_upload_data_object(client, file_like_object):
    """
    Upload an in-memory file to OpenAI and return the file ID.
   
    :param client: OpenAI client
    :param file_name: Name of the file (used for logging purposes only)
    :param file_like_object: BytesIO object containing the data
    :return: File ID
    """
    try:
        # Log the contents of the file
        file_contents = file_like_object.getvalue().decode('utf-8')
        print(f"Contents of in-memory file '{file_like_object.name}' before uploading:\n{file_contents}\n")

        # Reset the pointer to the start of the file
        file_like_object.seek(0)

        # Upload the file to OpenAI
        file_upload = client.files.create(file=file_like_object, purpose='assistants')

        # Check if the file already exists in the client's file list
        existing_files = list(client.files.list())
        for file in existing_files:
            print(f"file {file.filename} : {file.id}")


        return file_upload.id
    except Exception as e:
        print(f"Error uploading in-memory file: {e}")
        return None

    
def sist_create_vector_store(client, name, file_ids):
    """
    Wrapper function to create a vector store using the client.beta.vector_stores.create method.

    Parameters:
    - client: The client object to interact with the API.
    - name: The name of the vector store.
    - file_ids: List of file IDs to be included in the vector store.

    Returns:
    - response: The response from the client API call.
    """
    try:
        # Prepare the data for the API call
        data = {"name": name, "file_ids": file_ids}

        response = client.beta.vector_stores.create(**data)
        return response
    except Exception as e:
        print(f"An error occurred while creating the vector store: {str(e)}")
        raise


def sist_retrieve_vector_store(client, vector_store_id=None):
    """
    Wrapper function to retrieve a vector store using the client.beta.vector_stores.retrieve method.

    Parameters:
    - client: The ID of the vector store to retrieve.

    Returns:
    - response: The response from the client API call.
    """
    try:
        if vector_store_id is None:
            raise ValueError("vector_store_id cannot be None when retrieving a vector store.")
        
        response = client.beta.vector_stores.retrieve(
            vector_store_id=vector_store_id
        )
        return response
    except Exception as e:
        print(f"An error occurred while retrieving the vector store: {str(e)}")
        raise


def sist_create_and_check_vector_store(client, file_id=None):
    """
    Create a vector store with the uploaded file and wait until it's ready.
    
    Parameters:
    - client: The client object to interact with the API.
    - file_id: The ID of the file to associate with the vector store, or None.
    
    Returns:
    - vector_store_id: The ID of the created vector store.
    """
    if file_id is None:
        raise ValueError("file_id cannot be None when creating a vector store.")
    
    # Create the vector store with the provided file_id
    vector_store = sist_create_vector_store(client, "Weather Data", [file_id])
    vector_store_id = vector_store.id
    print(f"Vector store created with ID: {vector_store_id}")
    
    # Polling to check the status of the vector store creation
    while True:
        vector_store = sist_retrieve_vector_store(client, vector_store_id)
        if vector_store.status == 'completed':
            print("Vector store processing completed")
            break
        elif vector_store.status == 'failed':
            raise Exception("Vector store processing failed")
        time.sleep(1)
    
    return vector_store_id


#   ===========================================================
#   4. Thread and Message Handling

def sist_create_thread(client, messages, tool_resources=None):
    """
    Wrapper function to create a thread with optional file_search tool resources.

    Parameters:
    - client: The client object to interact with the API.
    - vector_store_id: The vector store ID to be used by the file_search tool (optional).

    Returns:
    - thread_id: The ID of the created thread.
    """

    try:
        print(f"create thread with {len(messages)} messages")
        print(f"create thread tool_resources: {tool_resources}")
        thread = client.beta.threads.create(
            messages=messages,
            tool_resources=tool_resources  # tool_resources passed for v2
        )
        return thread.id
    except Exception as e:
        print(f"An error occurred while creating the thread: {str(e)}")
        raise

def sist_print_run_details(run):
    print("\n--------------------------------------------")
    print(f"Run Details")
    print("--------------------------------------------")
    
    print(f"Run ID: {run.id}")
    print(f"Assistant ID: {run.assistant_id}")
    print(f"Created at: {run.created_at}")
    print(f"Expires at: {run.expires_at}")
    print(f"Status: {run.status}")
    
    if run.last_error:
        print(f"Last Error: {run.last_error}")
    else:
        print(f"No errors encountered.")
    
    print(f"Instructions:")
    print(run.instructions.strip())
    
    print(f"Tools:")
    for tool in run.tools:
        # Use dot notation for objects like CodeInterpreterTool
        print(f"  - Type: {tool.type}")

    print(f"\nParallel Tool Calls: {run.parallel_tool_calls}")
    print(f"Response Format: {run.response_format}")
    print("--------------------------------------------\n")

def sist_create_thread_run(client, thread_id, assistant_id, tools=[], stream=False, response_format="auto", model=None):
    """
    Run:
    A run is created on a thread, not on individual messages.
    When you create a run, the assistant processes all messages in the thread up to that point.
    The run generates a response based on the entire conversation context in the thread.

    Returns:
    - run_id: The ID of the created run.

    Run object
    {
    "id": "run_abc123",
    "object": "thread.run",
    "created_at": 1699075072,
    "assistant_id": "asst_abc123",
    "thread_id": "thread_abc123",
    "status": "completed",
    "started_at": 1699075072,
    "expires_at": null,
    "cancelled_at": null,
    "failed_at": null,
    "completed_at": 1699075073,
    "last_error": null,
    "model": "gpt-4o",
    "instructions": null,
    "incomplete_details": null,
    "tools": [
        {
        "type": "code_interpreter"
        }
    ],
    "tool_resources": {
        "code_interpreter": {
            "file_ids": [
                "file-abc123",
                "file-abc456"
            ]
        }
    },
    "metadata": {
        "user_id": "user_abc123"
    },
    "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579
    },
    "temperature": 1.0,
    "top_p": 1.0,
    "max_prompt_tokens": 1000,
    "max_completion_tokens": 1000,
    "truncation_strategy": {
        "type": "auto",
        "last_messages": null
    },
    "response_format": "auto",
    "tool_choice": "auto",
    "parallel_tool_calls": true
    }

    """
    try:
        run_params = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "stream": stream,
            "tools": tools,
            "tool_choice": "auto"
        }

        # Ensure the response format is valid, defaulting to "auto"
        if response_format == "auto":
            run_params["response_format"] = "auto"
        else:
            # Log an appropriate warning or handle alternative formats if needed
            print(f"Unsupported response format: {response_format}, defaulting to 'auto'.")
            run_params["response_format"] = "auto"

        run_params["tool_choice"] = "auto"
        run = client.beta.threads.runs.create(**run_params)
        sist_print_run_details(run)
        return run.id
    except Exception as e:
        print(f"An error occurred while creating the thread run: {str(e)}")
        raise


from pprint import pprint

def format_and_print_response(response):
    print("Tool Outputs Response:")
    print("=" * 40)  # Separator for readability

    # Access response attributes directly
    print(f"\nRun ID: {response.id}")
    print(f"Assistant ID: {response.assistant_id}")
    print(f"Thread ID: {response.thread_id}")
    print(f"Status: {response.status}")
    print(f"Created At: {response.created_at}")
    
    print("\nInstructions:")
    print(getattr(response, 'instructions', '').strip())

    # Display tools used
    print("\nTools Used:")
    tools = getattr(response, 'tools', [])
    for idx, tool in enumerate(tools, 1):
        tool_type = getattr(tool, 'type', None)
        function = getattr(tool, 'function', None)
        function_name = function.name if function else None

        print(f"  Tool {idx}:")
        print(f"    Type: {tool_type}")
        if function_name:
            print(f"    Function: {function_name}")

    # Display full tool outputs in a pretty format
    print("\nFull Tool Outputs:")
    pprint(vars(response), indent=2, width=100)
    print("=" * 40)

    # Further formatted step details
    steps = getattr(response, 'steps', [])
    for step in steps:
        print(f"\nProcessing step ID: {step['id']}")
        print(f"Step Type: {step['type']}")
        print(f"Step Status: {step['status']}")
        tool_calls = step.get('tool_calls', [])
        for call in tool_calls:
            print(f"  Tool Call ID: {call['id']}")
            print(f"  Tool Call Type: {call['type']}")
            if call.get('type') == 'function':
                print(f"    Function executed: {call['function']['name']}")
            elif call.get('type') == 'file_search':
                print(f"    File Search results: {call.get('file_search_results', [])}")
            elif call.get('type') == 'code_interpreter':
                print(f"    Code executed: {call.get('code', '')}")


def sist_submit_tool_outputs(client, thread_id, run_id, tool_outputs):
    """
    Wrapper function to submit tool outputs using the client.beta.threads.runs.submit_tool_outputs method.
    
    Parameters:
    - client: The client object to interact with the API.
    - thread_id: The ID of the thread where the outputs are to be submitted.
    - run_id: The ID of the run associated with the thread.
    - tool_outputs: The outputs to be submitted.
    
    Returns:
    - response: The response from the client API call.
    """
    try:
        # Ensure the tool_outputs are correctly structured as required by v2
        if not isinstance(tool_outputs, list) or not all(isinstance(output, dict) for output in tool_outputs):
            raise ValueError("tool_outputs must be a list of dictionaries")
        
        # Submit the tool outputs
        response = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )
        format_and_print_response(response)
        return response
    except Exception as e:
        print(f"An error occurred while submitting tool outputs: {str(e)}")
        raise


def sist_runs_steps_list(client, thread_id, run_id):
    """
    step object
    {
    "id": "step_abc123",
    "object": "thread.run.step",
    "created_at": 1699063291,
    "run_id": "run_abc123",
    "assistant_id": "asst_abc123",
    "thread_id": "thread_abc123",
    "type": "message_creation",
    "status": "completed",
    "cancelled_at": null,
    "completed_at": 1699063291,
    "expired_at": null,
    "failed_at": null,
    "last_error": null,
    "step_details": {
        "type": "message_creation",
        "message_creation": {
        "message_id": "msg_abc123"
        }
        },
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 456,
            "total_tokens": 579
        }
    }
    """

    return client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run_id)


def sist_runs_retrieve_status(client, thread_id, run_id):
    return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

#   ====================================
# 
def sist_get_required_args_from_spec(function_name: str, function_specs: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    print(f"sist_get_required_args_from_spec: {function_name}")
    for func_spec in function_specs:
        if func_spec["function"]["name"] == function_name:
            required = func_spec["function"]["parameters"].get("required", [])
            properties = func_spec["function"]["parameters"].get("properties", {})
            return required, properties
    raise ValueError(f"Function '{function_name}' not found in specifications.")


def sist_get_dynamic_output_schema(function_name: str) -> Dict:
    if function_name == "get_temperature":
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature": {"type": "number"},
                "status": {"type": "string"}
            },
            "required": ["city", "temperature", "status"],
            "additionalProperties": False
        }
    elif function_name == "calculate_difference":
        return {
            "type": "object",
            "properties": {
                "value1": {"type": "number"},
                "value2": {"type": "number"},
                "difference": {"type": "number"},
                "status": {"type": "string"}
            },
            "required": ["value1", "value2", "difference", "status"],
            "additionalProperties": False
        }
    return {}


def sist_manage_function_call(tool_call, tool_outputs, az_ent_impls, function_specs):
    try:
        # Extract function name and arguments from the tool_call
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        print(f"Executing function: {function_name} with arguments: {function_args}")
      
        if function_name not in az_ent_impls:
            print(f"Function '{function_name}' is not implemented.")
            return

        # Check if this tool call has already been processed
        if any(output["tool_call_id"] == tool_call.id for output in tool_outputs):
            print(f"Tool call {tool_call.id} already processed.")
            return  # Skip processing if it has already been handled

        # Get the function implementation
        function_impl = az_ent_impls[function_name]

        # Extract required arguments from the function specs
        required_args, properties = sist_get_required_args_from_spec(function_name, function_specs)

        # Validate and extract required arguments
        args_to_pass = {}
        for arg in required_args:
            if arg not in function_args:
                raise ValueError(f"Required argument '{arg}' not provided for function '{function_name}'")
            args_to_pass[arg] = function_args[arg]

        # Type checking for arguments
        for arg, value in args_to_pass.items():
            expected_type = properties[arg].get("type")
            if expected_type and not isinstance(value, (str, int, float)):
                raise TypeError(f"Argument '{arg}' should be of type {expected_type}, but got {type(value)}")

        # Execute the function with the extracted arguments
        function_result = function_impl(**args_to_pass)
        print(f"function_result: {function_result}")

        # **Add Schema Validation Here**
        # Get the output schema for this function
        schema = sist_get_dynamic_output_schema(function_name)

        # Validate the function result against the schema and append to tool_outputs
        sist_validate_and_assign_schema(tool_call, schema, tool_outputs, function_result)

        print(f"Function '{function_name}' executed successfully with result: {function_result}")
        
    except Exception as e:
        print(f"Error in sist_manage_function_call for function '{function_name}': {str(e)}")

#   ====================================
#   schema

def sist_validate_and_assign_schema(tool_call, schema, tool_outputs, function_result):
    """
    Validate the output of the tool call based on the dynamic schema and assign it.
    """
    try:
        # Validate the function result against the schema
        jsonschema.validate(instance=function_result, schema=schema)

        # If validation passes, assign the function result to the tool outputs
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": json.dumps(function_result)  # Store the result as a stringified JSON object
        })
        print(f"Validated and assigned schema for tool call {tool_call.id}")
    except jsonschema.exceptions.ValidationError as e:
        print(f"Schema validation failed for tool call {tool_call.id}: {str(e)}")
        raise e


def sist_manage_file_search(tool_call, tool_outputs):
    """
    Function to manage the file_search tool call.
    
    Parameters:
    - tool_call: The tool call object containing the file search query and results.
    - tool_outputs: The list to which tool outputs are appended.

    Returns:
    - None: Appends results to tool_outputs.
    """
    print(f"sist_manage_file_search with query: {tool_call.file_search.query}")

    # Ensure we handle the vector_store-based results in v2
    if tool_call.file_search.results:
        for result in tool_call.file_search.results:
            print(f"File search result: {result.content}")
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": json.dumps({"content": [r.content for r in tool_call.file_search.results]})
        })
    else:
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": json.dumps({"message": "No results found"})
        })


def sist_manage_code_interpreter(tool_call, tool_outputs):
    """
    Function to manage the code_interpreter tool call.
    
    Parameters:
    - tool_call: The tool call object containing the code interpreter input.
    - tool_outputs: The list to which tool outputs are appended.

    Returns:
    - None: Appends the completion message to tool_outputs.
    """
    print(f"sist_manage_code_interpreter with input: {tool_call.code_interpreter.input}")
    
    # Since execution is handled by OpenAI, just log completion
    tool_outputs.append({
        "tool_call_id": tool_call.id,
        "output": json.dumps({"message": "Code interpreter execution completed"})
    })


def show_run_step(step):
    """
    This function displays the details of a run step in a readable format.
    """
    try:
        step_type = step.type
        step_id = step.id
        assistant_id = step.assistant_id
        status = step.status
        created_at = step.created_at
        completed_at = step.completed_at if step.completed_at else "Not completed yet"
        thread_id = step.thread_id
        usage_info = step.usage if step.usage else "No usage data"
        
        print(f"Step ID: {step_id}")
        print(f"Assistant ID: {assistant_id}")
        print(f"Thread ID: {thread_id}")
        print(f"Type: {step_type}")
        print(f"Status: {status}")
        print(f"Created At: {created_at}")
        print(f"Completed At: {completed_at}")
        print(f"Usage Info: {usage_info}")

        # Show tool call details if available
        if step_type == 'tool_calls':
            tool_calls = step.step_details.tool_calls
            print(f"Tool Calls:")
            for idx, tool_call in enumerate(tool_calls, start=1):
                print(f"  Tool Call {idx}:")
                print(f"    ID: {tool_call.id}")
                print(f"    Type: {tool_call.type}")
                
                try:
                    # Check for function tool calls
                    if tool_call.type == "function":
                        # Handle function tool calls
                        function_call = tool_call.function
                        print(f"    Function Name: {function_call.name}")
                        print(f"    Function Arguments: {function_call.arguments}")
                        if function_call.output:
                            print(f"    Function Output: {function_call.output}")
                        else:
                            print(f"    Function Output: Not yet available")
                    
                    # Check for other tool types, e.g., file_search, code_interpreter
                    elif tool_call.type == "file_search":
                        print(f"    File Search - Ranking Options: {tool_call.file_search.ranking_options}")
                        print(f"    File Search Results: {tool_call.file_search.results}")
                    elif tool_call.type == "code_interpreter":
                        print(f"    Code Interpreter Input: {tool_call.code_interpreter.input}")
                        print(f"    Code Interpreter Outputs: {tool_call.code_interpreter.outputs}")

                except AttributeError as e:
                    print(f"Error: Missing attribute in tool call: {str(e)}")

        elif step_type == 'message_creation':
            message_creation = step.step_details.message_creation
            print(f"Message Created: ID: {message_creation.message_id}")
        
        print("\n------------------------------------\n")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        raise



def sist_run_thread_process(client, thread_id, run_id, az_ent_impls, az_ent_specs):
    tool_outputs = []
    tool_call_tracker = {}
    max_retries = 3
    retry_count = 0
    steps_list = []
    processed_step_ids = set()  # Track already processed step IDs
    total_steps_processed = 0  # Track the total number of steps processed so far
    print(f"\nSTEPS\n")

    try:
        while retry_count < max_retries:
            try:
                run_status = sist_runs_retrieve_status(client, thread_id, run_id)

                # Fetch the latest list of steps
                new_steps_list = list(sist_runs_steps_list(client, thread_id, run_id))

                # Process only new steps (those not already processed)
                for step_index, step in enumerate(new_steps_list):
                    if step.id not in processed_step_ids:
                        # Increment the total steps count to reflect the position
                        total_steps_processed += 1
                        print(f"Processing step {total_steps_processed} (new step {step_index + 1} of {len(new_steps_list)})")
                        show_run_step(step)
                        processed_step_ids.add(step.id)  # Mark this step as processed


                if run_status.status == "completed":
                    print("Run completed.")
                    break
                elif run_status.status == "failed":
                    print(f"Run failed. Error: {run_status.last_error}")
                    break
                elif run_status.status == "requires_action":
                    print("Requires Action.")
                    steps_list = list(sist_runs_steps_list(client, thread_id, run_id))
                    for step in steps_list:
                        print(f"Processing step: {step.type}, Status: {step.status}")
                        if step.type == "tool_calls" and step.status == "in_progress":
                            for tool_call in step.step_details.tool_calls:
                                call_id = tool_call.id

                                if tool_call_tracker.get(call_id, {}).get('executed', False):
                                    print(f"Tool call {call_id} already processed.")
                                    continue

                                # Process function calls
                                if tool_call.type == "function":
                                    sist_manage_function_call(tool_call, tool_outputs, az_ent_impls, az_ent_specs)

                                    # Gather data for final response
                                    function_result = json.loads(tool_outputs[-1]['output'])


                                    #   ========================    abstract code
    
                                    # Default value generator based on schema
                                    def get_default_value(arg_name, func_spec):
                                        """
                                        Generate default values based on the argument's type in the function schema.
                                        """
                                        # Retrieve the argument type from the function spec
                                        arg_properties = func_spec["function"]["parameters"]["properties"]
                                        
                                        if arg_name in arg_properties:
                                            arg_type = arg_properties[arg_name].get("type")
                                            
                                            # Generate default values based on the argument's type
                                            if arg_type == "string":
                                                return "default_string"
                                            elif arg_type == "number":
                                                return 0
                                            elif arg_type == "boolean":
                                                return False
                                            elif arg_type == "array":
                                                return []
                                            elif arg_type == "object":
                                                return {}
                                            else:
                                                return None  # Fallback in case of unrecognized types
                                        else:
                                            return None  # If the argument is not found in the schema

                                    # Function to process any function call dynamically
                                    def process_function_call(tool_call, function_result, az_ent_specs):
                                        """
                                        Process function results dynamically based on function specifications.
                                        """
                                        # Check if the function exists in the specs

                                        for spec in az_ent_specs:
                                            print(f"spec:");pprint(spec)

                                        if tool_call.function.name in [spec["function"]["name"] for spec in az_ent_specs]:
                                            # Get the function spec based on the function name
                                            func_spec = next(spec for spec in az_ent_specs if spec["function"]["name"] == tool_call.function.name)
                                            
                                            # Loop through the required arguments in the function spec
                                            for arg in func_spec["function"]["parameters"]["required"]:
                                                if arg not in function_result:
                                                    # Assign a default value if an argument is missing
                                                    function_result[arg] = get_default_value(arg, func_spec)

                                            # Now process the function result further based on the output of the function call
                                            for key, value in function_result.items():
                                                # Perform any custom logic based on the argument, like grouping the results
                                                if key not in tool_call_tracker:
                                                    tool_call_tracker[key] = {}
                                                tool_call_tracker[key][tool_call.function.name] = value



                                    # Generalize the function result handling
                                    process_function_call(tool_call, function_result, az_ent_specs)



                                # Handle file search calls
                                elif tool_call.type == "file_search":
                                    sist_manage_file_search(tool_call, tool_outputs)
                                    print(f"Processed file search tool call {call_id}")

                                # Handle other tool call types (e.g., code interpreter)
                                elif tool_call.type == "code_interpreter":
                                    sist_manage_code_interpreter(tool_call, tool_outputs)
                                    print(f"Processed code interpreter tool call {call_id}")


                                # Mark the tool call as executed
                                tool_call_tracker[call_id] = {'executed': True}

                    if tool_outputs:
                        print("Submitting tool outputs...")
                        sist_submit_tool_outputs(client, thread_id, run_id, tool_outputs)
                        tool_outputs.clear()
                        print("Tool outputs submitted successfully.")

                elif run_status.status == "in_progress":
                    print("Run is still in progress.")
                    time.sleep(5)

            except Exception as e:
                print(f"Error occurred in the inner loop: {e}")
                traceback.print_exc()  # Add traceback in the inner loop as well
                retry_count += 1
                time.sleep(5)

        if retry_count == max_retries:
            print("Max retries reached. Process failed.")
    
    except Exception as e:
        print(f"ERROR in outer loop: {str(e)}")
        traceback.print_exc()
        raise

# ===========================================================
# 8. Response Display

def sist_list_thread_messages(client, thread_id):
    """
    Wrapper function to list messages in a thread using the client.beta.threads.messages.list method.

    Parameters:
    - client: The client object to interact with the API.
    - thread_id: The ID of the thread from which to retrieve messages.

    Returns:
    - messages_iterable: The iterable of messages from the client API call.
    """
    try:
        messages_iterable = client.beta.threads.messages.list(
            thread_id=thread_id)
        return messages_iterable
    except Exception as e:
        print(f"An error occurred while listing the thread messages: {str(e)}")
        raise


def sist_get_assistant_response(client, thread_id):
    messages = sist_list_thread_messages(client, thread_id)
    
    # Loop through the messages to find the first assistant response
    for message in messages:
        if message.role == 'assistant':
            for content_block in message.content:
                assistant_response = content_block.text.value
                return assistant_response  # Return the first response and break out

    return None  # If no assistant response is found


# ===========================================================
# 9. Cleanup

def sist_cleanup_file(file_id, data_file):
    os.remove(data_file)
    response = sist_delete_file(openai, file_id)
    print(f"\n sist_delete_file response: {response}")
    print(f"Deleted file {data_file} with ID: {file_id}")


def sist_delete_file(openai_client, file_id):
    """
    Wrapper function to delete a file using the openai.files.delete method.

    Parameters:
    - openai_client: The OpenAI client object to interact with the API.
    - file_id: The ID of the file to be deleted.

    Returns:
    - response: The response from the OpenAI API.
    """
    try:
        response = openai_client.files.delete(file_id)
        return response
    except Exception as e:
        print(f"An error occurred while deleting the file: {str(e)}")
        raise

# ===========================================================

def sist_cleanup_all_resources(client):
    """
    Deletes all assistants, threads, vector stores, and files associated with the OpenAI account.
    Handles cases where resources might not exist.
    """
    try:
        # Delete Assistants
        print("Starting cleanup of assistants...")
        assistants = list(client.beta.assistants.list())  # Convert to list to prevent iteration issues
        for assistant in assistants:
            assistant_id = assistant.id
            try:
                client.beta.assistants.delete(assistant_id=assistant_id)
                print(f"Deleted assistant with ID: {assistant_id}")
            except openai.error.NotFoundError:
                print(f"Assistant with ID {assistant_id} not found. Skipping.")
            except Exception as e:
                print(f"Failed to delete assistant {assistant_id}: {e}")
                traceback.print_exc()

        # Delete Vector Stores
        print("\nStarting cleanup of vector stores...")
        vector_stores = list(client.beta.vector_stores.list())
        for vs in vector_stores:
            vector_store_id = vs.id
            try:
                client.beta.vector_stores.delete(vector_store_id=vector_store_id)
                print(f"Deleted vector store with ID: {vector_store_id}")
            except openai.error.NotFoundError:
                print(f"Vector store with ID {vector_store_id} not found. Skipping.")
            except Exception as e:
                print(f"Failed to delete vector store {vector_store_id}: {e}")
                traceback.print_exc()

        # Delete Files
        print("\nStarting cleanup of files...")
        files = list(client.files.list())
        for file in files:
            file_id = file.id
            try:
                client.files.delete(file_id)
                print(f"Deleted file with ID: {file_id}")
            except openai.error.NotFoundError:
                print(f"File with ID {file_id} not found. Skipping.")
            except Exception as e:
                print(f"Failed to delete file {file_id}: {e}")
                traceback.print_exc()

        # Optionally, delete local files if necessary
        try:
            os.remove("meteo_info.txt")
            print("\nDeleted local file 'meteo_info.txt'")
        except FileNotFoundError:
            print("\nLocal file 'meteo_info.txt' not found. Skipping.")
        except Exception as e:
            print(f"\nFailed to delete local file 'meteo_info.txt': {e}")
            traceback.print_exc()

        print("\nCleanup completed.")

    except Exception as e:
        print(f"An error occurred during cleanup: {e}")
        traceback.print_exc()



#   ===========================================================

#   Specific

#   ===========================================================


#   ===========================================================
#   Assistant

def sist_gen_assistant_name():
    return "whether_expert"

#   ===========================================================
#   File search


def sist_gen_data_file(type='text'):

    meteo_data = """
        city,humidity,pluviometry
        Madrid,57,35.25
        Zaragoza,62,26.83
        Barcelona,72,51
        Sevilla,61,44.5
    """

    meteo_info = """
    In Madrid, the average humidity level is 57%, and the average pluviometry (rainfall) is 35.25 mm per month
    In Zaragoza, the average humidity level is 62%, and the average pluviometry is 26.83 mm per month
    In Barcelona, the average humidity level is 72%, and the average pluviometry is 51 mm per month
    In Sevilla, the average humidity level is 61%, and the average pluviometry is 44.5 mm per month
    """

    if type == 'text':
        file_name = "meteo_info.txt"
        file_data = meteo_info.strip()
    else:
        file_name = "meteo_data.csv"
        file_data = meteo_data.strip()

    with open(file_name, "w") as f:
        f.write(file_data)

    return file_name, file_data
    

def sist_gen_data_object(type='text'):
    meteo_data = """
        city,humidity,pluviometry
        Madrid,57,35.25
        Zaragoza,62,26.83
        Barcelona,72,51
        Sevilla,61,44.5
    """

    meteo_info = """
    In Madrid, the average humidity level is 57%, and the average pluviometry (rainfall) is 35.25 mm per month
    In Zaragoza, the average humidity level is 62%, and the average pluviometry is 26.83 mm per month
    In Barcelona, the average humidity level is 72%, and the average pluviometry is 51 mm per month
    In Sevilla, the average humidity level is 61%, and the average pluviometry is 44.5 mm per month
    """

    if type == 'text':
        file_name = "meteo_info.txt"
        file_data = meteo_info.strip()
    else:
        file_name = "meteo_data.csv"
        file_data = meteo_data.strip()

    # Create a BytesIO object
    file_like_object = io.BytesIO(file_data.encode('utf-8'))
    file_like_object.name = file_name  # Add the 'name' attribute manually

    print(f"Generated in-memory file: {file_name}")
    return file_name, file_like_object



# ===========================================================
#   Function calling

def sist_get_specs():
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Fetch the current temperature for a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city to get the temperature for."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    return functions


def get_temperature(city: str) -> Dict[str, Union[str, float]]:
    mock_data = {
        "Madrid": 15.5,
        "Zaragoza": 16.5,
        "Barcelona": 16.5,
        "Sevilla": 19.2
    }
    return {
        "city": city,
        "temperature": mock_data.get(city, None),
        "status": "success" if city in mock_data else "error"
    }

def get_humidity(city: str) -> Dict[str, Union[str, float]]:
    mock_data = {
        "Madrid": 57,     
        "Zaragoza": 62,   
        "Barcelona": 72,  
        "Sevilla": 61     
    }
    
    return {
        "city": city,
        "humidity": mock_data.get(city, None),
        "status": "success" if city in mock_data else "error"
    }

def calculate_difference(value1: float, value2: float) -> Dict[str, Union[str, float]]:
    difference = abs(value1 - value2)
    return {
        "value1": value1,
        "value2": value2,
        "difference": difference,
        "status": "success"
    }


def sist_set_impls():
    impls = {
        "get_temperature": get_temperature
    }
    
    # Iterate over the key-value pairs in the impls dictionary
    for func_name, func_impl in impls.items():
        print(f"Function {func_name} registered. {type(func_impl)} : {func_impl}")
    
    return impls


def sist_get_impls():
    # Get the specifications
    specs = sist_get_specs()  # Ensure this returns valid function specs
    
    # Dictionary to store the implementations
    impls = {}
    
    # Get all functions in the current module
    current_module = sys.modules[__name__]
    other_module = sys.modules.get('module_with_functions', None)  # Example of another module to check

    # Get functions from the current module
    module_functions = {name: obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)}
    
    # Optionally, update with functions from another module if needed
    if other_module:
        module_functions.update({name: obj for name, obj in inspect.getmembers(other_module, inspect.isfunction)})

    # Iterate through the specifications
    for func_spec in specs:
        func_name = func_spec['function']['name']
        print(f"Checking for function: {func_name}")
        
        # Check if the function exists in the retrieved functions
        if func_name in module_functions:
            impls[func_name] = module_functions[func_name]

    # Iterate over the key-value pairs in the impls dictionary
    for func_name, func_impl in impls.items():
        print(f"Function {func_name} registered. {type(func_impl)} : {func_impl}")
    


    return impls



#   ===========================================================
#   Schemas

def sist_get_response_format():

    response_format = {
    "cities": {
        "city1": {
        "name": "Name of the first city",
        "temperature": "Temperature value in degrees",
        "humidity": "Humidity percentage for city1"
        },
        "city2": {
        "name": "Name of the second city",
        "temperature": "Temperature value in degrees",
        "humidity": "Humidity percentage for city2"
        }
    },
    "differences": {
        "temperature": {
        "value": "Difference in temperature between city1 and city2",
        "comparison": "Textual comparison between city1 and city2 temperatures",
        "city1_name": "Name of the first city",
        "city2_name": "Name of the second city"
        },
        "humidity": {
        "value": "Difference in humidity between city1 and city2",
        "comparison": "Textual comparison between city1 and city2 humidities",
        "city1_name": "Name of the first city",
        "city2_name": "Name of the second city"
        }
    },
    "assistant_thoughts": "Assistant's conclusion or summary about the temperature and humidity comparison"
    }
    return response_format


def sist_get_output_schema():

    output_schema = {
        "type": "object",
        "properties": {
            "cities": {
            "type": "object",
            "properties": {
                "city1": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "temperature": { "type": "number" },
                    "humidity": { "type": "number" }
                },
                "required": ["name", "temperature", "humidity"]
                },
                "city2": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "temperature": { "type": "number" },
                    "humidity": { "type": "number" }
                },
                "required": ["name", "temperature", "humidity"]
                }
            },
            "required": ["city1", "city2"]
            },
            "differences": {
            "type": "object",
            "properties": {
                "temperature": {
                "type": "object",
                "properties": {
                    "value": { "type": "number" },
                    "comparison": { "type": "string" },
                    "city1_name": { "type": "string" },
                    "city2_name": { "type": "string" }
                },
                "required": ["value", "comparison", "city1_name", "city2_name"]
                },
                "humidity": {
                "type": "object",
                "properties": {
                    "value": { "type": "number" },
                    "comparison": { "type": "string" },
                    "city1_name": { "type": "string" },
                    "city2_name": { "type": "string" }
                },
                "required": ["value", "comparison", "city1_name", "city2_name"]
                }
            },
            "required": ["temperature", "humidity"]
            },
            "assistant_thoughts": {
            "type": "string"
            }
        },
        "required": ["cities", "differences", "assistant_thoughts"]
    }

    return output_schema


#   ===========================================================
#   Instructions

def sist_get_goal():
    """
    Define the goal for the assistant.
    """
    goal = "provide weather data, specifically temperatures, humidities, and their differences for the requested cities."
    return goal

def sist_get_tasks():
    """
    Define the tasks for the assistant to follow.
    """
    tasks = [
        "First, check if temperature data is available by calling the **get_temperature** function.",

        "Use the file_search tool to find the humidity data for the specified cities from the uploaded file."

        "Use the Code Interpreter tool for any arithmetic calculations, such as calculating differences between values."
    ]
    return tasks


#   ===========================================================
#   user message

def sist_get_usr_msgs(response_format=None, functions=None):
    """
    Generates a list of user messages. The first message is the user's content,
    and the second message optionally includes instructions for the response format.
    
    Parameters:
    - response_format: An optional parameter to specify the format the assistant should use to respond.
    
    Returns:
    - A list of messages.
    """

    # usr_content = "Please respond with temperature and humidity differences directly as JSON without any additional wrapping like properties."

    usr_content = "Please find temperature and humidity values and indicate their difference between Madrid and Zaragoza."
    
    # First message: user request
    messages = [
        {"role": "user", "content": usr_content}
    ]
    
    # If a response format is specified, add a second message for response formatting
    if response_format:
        response_format = json.dumps(response_format)
        response_format_msg = {
            "role": "user",
            "content": f"Please respond with a raw JSON object that strictly complies with the following schema: {response_format}. Do not use Markdown formatting or code blocks. The response should be a valid JSON object that can be directly parsed."
        }
        messages.append(response_format_msg)
    
    # If a response format is specified, add a second message for response formatting
    if functions:
        custom_functions = json.dumps(response_format)
        custom_functions_msg = {
            "role": "user",
            "content": f"Please note the following functions: {custom_functions}."
        }
        messages.append(custom_functions_msg)
        
    return messages

# =============================================



# =============================================
def main():
    parser = argparse.ArgumentParser(description="Assistant Script")
    parser.add_argument('--reset', action='store_true', help='Reset and delete all created resources')
    parser.add_argument('-m', '--model', type=str, help='Specify the model to be used for the assistant')
    args = parser.parse_args()

    # Settings that can be configured dynamically
    upload_files = True  # Updated to True to upload files
    use_file_search = True
    use_code_interpreter = True
    use_function_calling = True
    use_output_schema = True

    # =============================================
    # 1. API Setup
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 1: API Setup")
    print("--------------------------------------------")

    client = sist_get_client()  # Initialize the OpenAI client

    if args.reset:
        print("Resetting and deleting all resources...")
        sist_cleanup_all_resources(client)  # Cleanup if reset argument is provided
        print("All resources have been reset and deleted.")
        return

    # Fetch the model from arguments
    selected_model = sist_get_model(args.model)
    print(f"Using model: {selected_model}")

    # =============================================
    # 2. Response Format
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 2: Output Schema")
    print("--------------------------------------------")

    response_format = None
    output_schema = None

    if use_output_schema:
        output_schema = sist_get_output_schema()
        schema_keys = list(output_schema.get("properties", {}).keys())
        print(f"Top-level elements of the response schema: {schema_keys}")


        response_format = sist_get_response_format()  # Fetch your custom schema
        format_keys = list(response_format.keys())
        print(f"Top-level elements of the response format: {format_keys}")

    # =============================================
    # 3. File Management
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 3: File Management")
    print("--------------------------------------------")

    if upload_files:
        file_id = None
        # Generate in-memory file data
        file_name, file_data = sist_gen_data_file()
        print(f"file_name: {file_name}")
        
        # Check for existing files
        existing_files = list(client.files.list())

        for file in existing_files:
            if file.filename == file_name:
                file_id = file.id
                print(f"Using existing file with ID: {file_id}")
                break
        
        if not file_id:
            # Upload the in-memory file to OpenAI
            file_id = sist_upload_data_file(client, file_name)
            if file_id:
                print(f"Uploaded new in-memory file with ID: {file_id}")
            else:
                print("Failed to upload in-memory file. Exiting.")

    # =============================================
    # 4. Vector Store Creation
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 4: Vector Store Creation")
    print("--------------------------------------------")

    print(f"use_file_search = {use_file_search}")

    vector_store_id = None
    if use_file_search:
        vector_stores = list(client.beta.vector_stores.list())
        for vs in vector_stores:
            if vs.name == "Weather Data":
                vector_store_id = vs.id
                print(f"Using existing vector store with ID: {vector_store_id}")
                break

        if not vector_store_id:
            vector_store_id = sist_create_and_check_vector_store(client, file_id)
            print(f"Created new vector store with ID: {vector_store_id}")

    # =============================================
    # 5. Function Loading and Instructions
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 5: Function Loading and Instructions")
    print("--------------------------------------------")

    az_ent_impls = sist_get_impls()  # Load function implementations
    az_ent_specs = sist_get_specs()  # Load custom functions
    print(f"az_ent_specs:");pprint(az_ent_specs)
    tools = []
    if use_function_calling:
        for spec in az_ent_specs:
            tools.append(spec)


    # =============================================
    # 6. Assistant Creation
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 6: Assistant Creation")
    print("--------------------------------------------")

    assistant_name = sist_gen_assistant_name()
    iz_assistant_id = sist_find_assistant(assistant_name)
    goal = sist_get_goal()
    tasks = sist_get_tasks()
    az_instructions = sist_get_instructions(goal, tasks, az_ent_specs, use_file_search, use_code_interpreter, output_schema)

    if not iz_assistant_id:
        iz_assistant_id = sist_create_assistant(
            client,
            instructions=az_instructions,
            tools=tools,
            model=selected_model  # Use selected model for the assistant
        )
        print(f"Assistant created with ID: {iz_assistant_id}")
    else:
        print(f"Using existing assistant with ID: {iz_assistant_id}")

    # =============================================
    # 7. User Messages and Response Format
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 7: User Messages and Response Format")
    print("--------------------------------------------")


    tz_usr_msgs = sist_get_usr_msgs(response_format=response_format, functions=None)

    print("Generated user messages:")
    for i, msg in enumerate(tz_usr_msgs):
        print(f"Message {i + 1}:")
        print(f"  Role: {msg['role']}")
        print(f"  Content: {msg['content'][:100]}...")

    # =============================================
    # 8. Tools
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 8: Tools")
    print("--------------------------------------------")
    # https://platform.openai.com/docs/api-reference/runs/createRun

    run_tools = []
    tool_resources = {}

    if use_function_calling:
        run_tools = tools

    if use_code_interpreter:
        print(f"append code_interpreter to tools")
        run_tools.append(
            {"type": "code_interpreter"}
        )
    else:
        print("Code interpreter not available.") 

    if use_file_search and vector_store_id is not None:
        print(f"append file_search to tools")
        run_tools.append(
            {"type": "file_search"}
        )
        print(f"add vector_store {vector_store_id} to file_search") 
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }           
    else:
        print("File search and vector store not available.")


    # Final output showing the tools and resources
    print("\nFinal tool setup:")
    print(f"Tools to be used: {run_tools}")
    if tool_resources:
        print(f"Tool resources: {tool_resources}")
    else:
        print("No tool resources defined.")
        
    # =============================================
    # 9. Thread Creation
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 9: Thread Creation")
    print("--------------------------------------------")

    wz_thread_id = sist_create_thread(
        client, 
        tz_usr_msgs,
        tool_resources=tool_resources
    )
    print(f"Thread created with ID: {wz_thread_id}")

    # =============================================
    # 10. Run Execution
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 10: Run Execution")
    print("--------------------------------------------")

    tz_run_id = sist_create_thread_run(
        client,
        thread_id=wz_thread_id,
        assistant_id=iz_assistant_id,
        stream=False,
        model=selected_model,
        tools=run_tools
    )
    print(f"Run started with run ID: {tz_run_id}")

    # =============================================
    # 11. Process the Run
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 11: Processing the Run")
    print("--------------------------------------------")

    sist_run_thread_process(client, wz_thread_id, tz_run_id, az_ent_impls, az_ent_specs)

    # =============================================
    # 12. Response Display and Validation
    # =============================================
    print("\n--------------------------------------------")
    print("BIT 12: Displaying and validating the assistant's response...")
    print("--------------------------------------------")

    # Get assistant's response
    assistant_response = sist_get_assistant_response(client, wz_thread_id)
    print(f"Assistant's response:"); pprint(assistant_response)

    try:
        # Attempt to parse the response as JSON
        parsed_response = json.loads(assistant_response)

        # Validate parsed JSON against the output schema
        jsonschema.validate(instance=parsed_response, schema=output_schema)
        print("\n\nFinal response validated successfully.")

    except json.JSONDecodeError:
        # Handle the case where the response is not a valid JSON
        print("Error: The assistant response is not a valid JSON.")
        print(f"Received response: {assistant_response}")

    except jsonschema.ValidationError as e:
        # Handle schema validation errors
        print("Error: The assistant response JSON did not match the expected schema.")
        print(f"Validation error details: {e}")

    except Exception as e:
        # Catch-all for any other unexpected exceptions
        print("An unexpected error occurred.")
        print(f"Error details: {e}")


if __name__ == "__main__":
    main()
