# work flow of the pipeline

![MAIN PROCESS LAMA-8B (3)](https://github.com/user-attachments/assets/813e1345-ecb9-425f-b767-df9848234cd5)

## graph TD
    A[Input] --> B{Input Type}
    B -->|Text| C[Language Detection]
    B -->|Audio| D[Speech-to-Text]
    D --> C
    C --> E[Translation to English]
    E --> F[Llama-3.1-8B Processing]
    F --> G{Output Format}
    G -->|Text| H[Translation to Indian Language]
    G -->|Audio| I[Text-to-Speech]
    H --> J[Output]
    I --> J

## Key Components & File Structure:

         pipeline/
      ├── main.py              # Main orchestration script
      ├── configs/
      │   ├── ds_offload.json  # Deepspeed NVMe config
      │   └── models.yaml      # Model configurations
      ├── utils/
      │   ├── language_utils.py
      │   └── ds_wrappers.py
      └── requirements.txt


  
