# Common Use Case Process
- Research: Documents, Internet, CRM
- Analysis: Compare, Extract, Infer
- Summary: Learning, Charts, Exec Summary
- Reporting: PDF, JSON, Markdown

# Process Types
- Sequential
- Hierarchical
- Hybrid
- Parallel
- Async
- Multi-Crews and Flows

# PWC Notes
- Agent and Task developemnt patterns
- Focus on accuracy and user experience
- Start Simple then More Complex: Crawl, walk, run, fly
- Basic prompt engineering, RAG pipelines


# Using Config Files
    import yaml
    from crewai import Agent, Task, Crew, LLM

    # Define file paths for YAML configurations
    files = {"agents": "config/agents.yaml", "tasks": "config/tasks.yaml"}
    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, "r") as file:
            configs[config_type] = yaml.safe_load(file)
    # Assign loaded configurations to specific variables
    agents_config = configs["agents"]
    tasks_config = configs["tasks"]

    # Creating Agents
    project_planning_agent = Agent(config=agents_config["project_planning_agent"])

    # Creating Tasks
    task_breakdown = Task(
        config=tasks_config["task_breakdown"], agent=project_planning_agent
    )

# Load the LLM Model
    llm_model = LLM(model="ollama/llama3.2:3b", temperature=0.5)

# Pydantic Models for Structured Output
    from typing import List
    from pydantic import BaseModel, Field

    class TaskEstimate(BaseModel):
        task_name: str = Field(..., description="Name of the task")
        estimated_time_hours: float = Field(
            ..., description="Estimated time to complete the task in hours"
        )
        required_resources: List[str] = Field(
            ..., description="List of resources required to complete the task"
        )

    class Milestone(BaseModel):
        milestone_name: str = Field(..., description="Name of the milestone")
        tasks: List[str] = Field(
            ..., description="List of task IDs associated with this milestone"
        )

    class ProjectPlan(BaseModel):
        tasks: List[TaskEstimate] = Field(
            ..., description="List of tasks with their estimates"
        )
        milestones: List[Milestone] = Field(..., description="List of project milestones")

    resource_allocation = Task(
        config=tasks_config["resource_allocation"],
        agent=resource_allocation_agent,
        output_pydantic=ProjectPlan,  # This is the structured output we want
    )

# Human Readable Crew Input
    from IPython.display import display, Markdown

    project = "Website"
    industry = "Technology"
    project_objectives = "Create a website for a small business"
    team_members = """
        - John Doe (Project Manager)
        - Jane Doe (Software Engineer)
    """
    project_requirements = """
        - Create a responsive design that works well on desktop and mobile devices
        - Implement a modern, visually appealing user interface with a clean look
    """

    # Format the dictionary as Markdown for a better display in Jupyter Lab
    formatted_output = f"""
    **Project Type:** {project}
    **Project Objectives:** {project_objectives}
    **Industry:** {industry}
    **Team Members:**
        {team_members}
    **Project Requirements:**
        {project_requirements}
    """
    # Display the formatted output as Markdown
    display(Markdown(formatted_output))

# Usage Metrics and Costs
    costs = (
        0.150 / 1_000_000 # Cost per token
        * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens)
    )
    print(f"Total costs: ${costs:.4f}")

# Create Custom Tools
    from crewai_tools import BaseTool

    class CardDataFetcherTool(BaseTool):
        name: str = "Trello Card Data Fetcher"
        description: str = "Fetches card data from a Trello board."

        def _run(self, card_id: str) -> dict:
            """
            Retrieves the details of a Trello card based on the provided card ID.

            Args:
                card_id (str): The ID of the Trello card to retrieve.

            Returns:
                dict: A dictionary containing the details of the Trello card in JSON format.
            """
            if card_id == "66c3bfed69b473b8fe9d922e":
                return json.dumps(
                    [
                        {
                            "id": "66c3bfed69b473b8fe9d922e",
                            "badges": {
                                ...
                            }
                        }
                    ]
                )

    print(
        json.dumps(
            json.loads(CardDataFetcherTool()._run("66c3bfed69b473b8fe9d922e")), indent=4
        )
    )

    # Creating Agents
    data_collection_agent = Agent(
        config=agents_config["data_collection_agent"],
        tools=[BoardDataFetcherTool(), CardDataFetcherTool()],
    )

# Multiple Crews
    # Define file paths for YAML configurations
    files = {
        "lead_agents": "config/lead_qualification_agents.yaml",
        "lead_tasks": "config/lead_qualification_tasks.yaml",
        "email_agents": "config/email_engagement_agents.yaml",
        "email_tasks": "config/email_engagement_tasks.yaml",
    }

    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, "r") as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    lead_agents_config = configs["lead_agents"]
    lead_tasks_config = configs["lead_tasks"]
    email_agents_config = configs["email_agents"]
    email_tasks_config = configs["email_tasks"]

# Basic Flow with Branching & Iteration
    from crewai import Flow
    from crewai.flow.flow import listen, start

    class SalesPipeline(Flow):
        @start()
        def fetch_leads(self):
            # Pull our leads from the database
            leads = [
                {
                    "lead_data": {
                        "name": "João Moura",
                        ...
                    },
                },
            ]
            return leads

        @listen(fetch_leads)
        def score_leads(self, leads):
            scores = lead_scoring_crew.kickoff_for_each(leads)
            self.state["score_crews_results"] = scores
            return scores

        @listen(score_leads)
        def store_leads_score(self, scores):
            # Here we would store the scores in the database
            return scores

        @listen(score_leads)
        def filter_leads(self, scores):
            return [score for score in scores if score["lead_score"].score > 70]

        @listen(filter_leads)
        def write_email(self, leads):
            scored_leads = [lead.to_dict() for lead in leads]
            emails = email_writing_crew.kickoff_for_each(scored_leads)
            return emails

        @listen(write_email)
        def send_email(self, emails):
            # Here we would send the emails to the leads
            return emails


    flow = SalesPipeline()

# Complex Flow with And, Or and Routing
    from crewai import Flow
    from crewai.flow.flow import listen, start, and_, or_, router

    class SalesPipeline(Flow):

        @start()
        def fetch_leads(self):
            leads = [
                {
                    "lead_data": {
                        "name": "João Moura",
                        ...
                    },
                },
            ]
            return leads

        @listen(fetch_leads)
        def score_leads(self, leads):
            scores = lead_scoring_crew.kickoff_for_each(leads)
            self.state["score_crews_results"] = scores
            return scores

        @listen(score_leads)
        def store_leads_score(self, scores):
            # Here we would store the scores in the database
            return scores

        @listen(score_leads)
        def filter_leads(self, scores):
            return [score for score in scores if score["lead_score"].score > 70]

        @listen(and_(filter_leads, store_leads_score))
        def log_leads(self, leads):
            print(f"Leads: {leads}")

        @router(filter_leads)
        def count_leads(self, scores):
            if len(scores) > 10:
                return "high"
            elif len(scores) > 5:
                return "medium"
            else:
                return "low"

        @listen("high")
        def store_in_salesforce(self, leads):
            return leads

        @listen("medium")
        def send_to_sales_team(self, leads):
            return leads

        @listen("low")
        def write_email(self, leads):
            scored_leads = [lead.to_dict() for lead in leads]
            emails = email_writing_crew.kickoff_for_each(scored_leads)
            return emails

        @listen(write_email)
        def send_email(self, emails):
            # Here we would send the emails to the leads
            return emails

# Testing our Crew
    support_report_crew.test(n_iterations=1, openai_model_name=llm_model.model)

# Training your crew and agents
    support_report_crew.train(n_iterations=1, filename="training.pkl")

# CLI
    crewai create crew <project>

    crewai run
    crewai train
    crewai replay
    crewai test

    # Creates an API
    crewai deploy