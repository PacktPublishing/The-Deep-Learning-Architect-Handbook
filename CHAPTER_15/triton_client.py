import tritonclient.http as httpclient


MODEL_NAME = "transformer_tensorrt_text_generation"
MODEL_VERSION = "1"

def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")
    input_text = np.array(["Tell me a joke."], dtype=object)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("TEXT", (1,), datatype="BYTES")
    ]
    input_tensors[0].set_data_from_numpy(input_text)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT_TEXT")
    ]
    query_response = client.infer(model_name=MODEL_NAME,
                                  model_version=MODEL_VERSION,
                                  inputs=input_tensors,
                                  outputs=outputs)
    output_text = query_response.as_numpy("OUTPUT_TEXT")
    print(output_text)

if __name__ == '__main__':
    main()
