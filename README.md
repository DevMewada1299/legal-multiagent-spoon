# Legal Document Analysis with Multi-Agent AI

This project is a Streamlit web application that uses a multi-agent AI system to analyze legal documents. Users can upload a PDF, and the application will identify and categorize key risks and obligations within the text, providing a comprehensive summary and a clause-by-clause analysis.

For trying it out please go to this link for a sample document and a key : https://drive.google.com/drive/folders/1HMsPRjaaqmop1GLCFYcn6p5q3CN31gzu?usp=sharing (Prototype)

Special Thanks to TRAE for providing an AI-Native IDE, accelerating our throuphput by 3x :) 

## Features

*   **PDF Document Upload**: Easily upload legal documents in PDF format.
*   **AI-Powered Analysis**: Leverages a multi-agent system to perform in-depth analysis of the document.
*   **Risk and Obligation Identification**: Automatically identifies and categorizes potential risks and obligations.
*   **Executive Summary**: Generates a concise, easy-to-read executive summary of the document.
*   **Interactive UI**: A user-friendly interface built with Streamlit, featuring a tabbed layout for easy navigation.
*   **Clause Explorer**: Allows users to explore individual clauses and their associated risks and obligations.

## How it Works: The `spoon-ai` Integration

This project is built on the `spoon-ai` SDK, which provides a powerful framework for creating and managing multi-agent AI systems. Here's a detailed look at how `spoon-ai` is used in this application:

### 1. The Graph-Based Pipeline

The core of the application is a graph-based pipeline defined in `graph_pipeline/graph.py`. This pipeline orchestrates the different stages of the document analysis process. The graph is built using `spoon-ai`'s `DeclarativeGraphBuilder`, which allows for a clear and concise definition of the pipeline's structure.

The `DeclarativeGraphBuilder` simplifies the process of creating complex, stateful AI workflows. Instead of writing boilerplate code to manage the flow of data and the execution of different components, you can define the pipeline's structure in a declarative way. This makes the code easier to read, understand, and maintain.

Here's how it works:

*   **`NodeSpec`**: Each step in the pipeline is defined as a `NodeSpec`. This includes a name for the node and a handler function that contains the business logic for that step.
*   **`EdgeSpec`**: The connections between the nodes are defined using `EdgeSpec`. This determines the order in which the nodes are executed.
*   **`GraphTemplate`**: The `NodeSpec` and `EdgeSpec` objects are combined into a `GraphTemplate`, which provides a complete blueprint of the pipeline.

The `DeclarativeGraphBuilder` takes this template and compiles it into a `StateGraph` from the `spoon-ai` SDK. The `StateGraph` is responsible for managing the application's state as it moves through the pipeline, ensuring that data is passed seamlessly from one node to the next.

The graph in this application consists of two main nodes:

*   **`extract_clauses`**: This node is responsible for extracting individual clauses from the legal document.
*   **`full_analysis`**: This node performs the main analysis, identifying risks and obligations within the extracted clauses.

### 2. Custom AI Agents

The analysis is performed by a set of custom AI agents, each with a specific role. These agents are defined in `graph_pipeline/agents.py` and inherit from `spoon-ai`'s `BaseAgent` (or a custom base agent that extends it).

*   **`ClauseExtractionAgent`**: This agent is responsible for the initial processing of the legal text, breaking it down into individual clauses.
*   **`ComprehensiveClauseAnalyserAgent`**: This is the core analysis agent. It takes the extracted clauses and uses a powerful language model to identify and categorize risks and obligations.
*   **`SummarizationAgent`**: This agent generates the executive summary of the document.

### 3. Language Model Interaction

The agents interact with the Gemini language model through the `ChatBot` class from the `spoon-ai` SDK. This class provides a simple and convenient way to send prompts to the language model and receive responses. The `ChatBot` class handles the complexities of API interaction, allowing the agents to focus on their specific tasks.

## Getting Started

### Prerequisites

*   Python 3.8+
*   A Gemini API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DevMewada1299/legal-multiagent-spoon.git
    cd legal-multiagent-spoon
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add your Gemini API key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

### Running the Application

To run the application locally, use the following command:

```bash
streamlit run app/ui/streamlit_app.py
```

## Deployment

This application is ready to be deployed to Streamlit Community Cloud.
