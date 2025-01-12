# Basic Agent
    planner = Agent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory=(
            "You're working on planning a blog article "
            ...
        ),
        allow_delegation=False,
        verbose=True,
        max_iter=2,  # Default: 20 iterations
    )

# Basic Task
    plan = Task(
        description=(
            "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
            ...
        ),
        expected_output=(
            "A comprehensive content plan document "
            ...
        ),
        agent=planner,
    )

# Basic Crew
    from crewai import Crew, LLM

    llm_model = "ollama/llama3.1:8b"

    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=True,
        manager_llm=LLM(
            model=llm_model, temperature=0.7, api_base="http://localhost:11434"
        ),
    )
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

# Tools
    from crewai_tools import (
        ScrapeWebsiteTool,
        FileReadTool,
        MDXSearchTool,
        SerperDevTool,
        DirectoryReadTool,
    )

    docs_scrape_tool = ScrapeWebsiteTool(
        website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
    )
    scrape_tool = ScrapeWebsiteTool()
    read_resume = FileReadTool(file_path='./fake_resume.md')
    csv_tool = FileReadTool(file_path="./support_tickets_data.csv")
    file_read_tool = FileReadTool()
    semantic_search_resume = MDXSearchTool(mdx='./fake_resume.md')
    search_tool = SerperDevTool()
    directory_read_tool = DirectoryReadTool(directory="./instructions")

    researcher = Agent(
        role="Tech Job Researcher",
        goal=(
            "Make sure to do amazing analysis on "
            ...
        ),
        backstory=(
            "As a Job Researcher, your prowess in "
            ...
        )
        tools = [scrape_tool, search_tool],
        verbose=True,
    )

    personalized_outreach_task = Task(
        description=(
            "Using the insights gathered from "
            ...
        ),
        expected_output=(
            "A series of personalized email drafts "
            "tailored to {lead_name}, "
        ),
        tools=[sentiment_analysis_tool, search_tool],
        agent=lead_sales_rep_agent,
    )

# Custom Tool

    from crewai_tools import BaseTool

    class SentimentAnalysisTool(BaseTool):
        name: str = "Sentiment Analysis Tool"
        description: str = (
            "Analyzes the sentiment of text "
            "to ensure positive and engaging communication."
        )

        def _run(self, text: str) -> str:
            # Your custom code tool goes here
            return "positive"

    sentiment_analysis_tool = SentimentAnalysisTool()

- Create a custom tool using crewAi's [BaseTool](https://docs.crewai.com/core-concepts/Tools/#subclassing-basetool) class
- Every Tool needs to have a `name` and a `description`.
- For simplicity and classroom purposes, `SentimentAnalysisTool` will return `positive` for every text.
- When running locally, you can customize the code with your logic in the `_run` function.


# Memory
    crew = Crew(
        agents=[support_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, quality_assurance_review],
        verbose=True,
        memory=True,
    )

# Using Pydantic Objects, JSON Output & Human Input
    from pydantic import BaseModel

    class VenueDetails(BaseModel):
        name: str
        address: str
        capacity: int
        booking_status: str

    venue_task = Task(
        description="Find a venue in {event_city} "
            "that meets criteria for {event_topic}.",
        expected_output="All the details of a specifically chosen"
            "venue you found to accommodate the event.",
        human_input=True,
        output_json=VenueDetails,
        output_file="venue_details.json",
        # Outputs the venue details as a JSON file
        agent=venue_coordinator,
    )

- Create a class `VenueDetails` using [Pydantic BaseModel](https://docs.pydantic.dev/latest/api/base_model/).
- Agents will populate this object with information about different venues by creating different instances of it.
- By using `output_json`, you can specify the structure of the output you want.
- By using `output_file`, you can get your output in a file.
- By setting `human_input=True`, the task will ask for human feedback (whether you like the results or not) before finalising it.

# Async Execution
    research_task = Task(
        description=(
            "Analyze the job posting URL provided ({job_posting_url}) "
            ...
        ),
        expected_output=(
            "A structured list of job requirements, including necessary "
            "skills, qualifications, and experiences."
        ),
        agent=researcher,
        async_execution=True
    )
- By setting `async_execution=True`, it means the task can run in parallel with the tasks which come after it.

# Hierarchy & Managers
    from crewai import Crew, Process, LLM

    # Define the crew with agents and tasks
    financial_trading_crew = Crew(
        agents=[
            data_analyst_agent,
            trading_strategy_agent,
            execution_agent,
            risk_management_agent,
        ],
        tasks=[
            data_analysis_task,
            strategy_development_task,
            execution_planning_task,
            risk_assessment_task,
        ],
        manager_llm=LLM(
            model=llm_model, temperature=0.7, api_base="http://localhost:11434"
        ),
        process=Process.hierarchical,
        verbose=True,
    )

- The `Process` class helps to delegate the workflow to the Agents (kind of like a Manager at work)
- In the example below, it will run this hierarchically.
- `manager_llm` lets you choose the "manager" LLM you want to use.

# Context
    resume_strategy_task = Task(
        description=(
            "Using the profile and job requirements obtained from "
            ...
        ),
        expected_output=(
            "An updated resume that effectively highlights the candidate's "
            "qualifications and experiences relevant to the job."
        ),
        output_file="tailored_resume.md",
        context=[research_task, profile_task],
        agent=resume_strategist
    )

- You can pass a list of tasks as `context` to a task.
- The task then takes into account the output of those tasks in its execution.
- The task will not run until it has the output(s) from those tasks.