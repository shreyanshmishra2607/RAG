import os
from crewai_tools import RagTool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class LatestAiDevelopment():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Load model config from environment with correct provider format
    model = os.getenv("MODEL", "ollama/llama3.1")  # Correctly formatted with provider
    api_base = os.getenv("API_BASE", "http://localhost:11434")

    # Custom configuration with explicit provider information
    rag_config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1",  # Provider is explicit in config
                "base_url": api_base
            }
        },
        "embedding_model": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "base_url": api_base
            }
        }
    }

    # Initialize the RAG tool with Ollama config
    rag_tool = RagTool(config=rag_config, summarize=False)

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[self.rag_tool],
            verbose=True,
            llm_config={
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),  # Remove provider prefix if present
                "base_url": self.api_base
            }
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            llm_config={
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),  # Remove provider prefix if present
                "base_url": self.api_base
            }
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            llm_config={
                "provider": "ollama",
                "model": self.model.replace("ollama/", ""),  # Remove provider prefix if present
                "base_url": self.api_base
            }
        )