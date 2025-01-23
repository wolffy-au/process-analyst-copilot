import json
from crewai import Flow, Crew
from crewai.flow.flow import listen, start
from process_analyst_copilot import ClarifyTheAsk


class ProcessPipeline(Flow):
    @start()
    def draft_process(self, clarify_the_ask: ClarifyTheAsk, input_ask: str):
        clarify_the_ask.setup_bpa_agent()
        clarify_the_ask.setup_draft_process()
        crew = Crew(
            agents=[clarify_the_ask.business_process_analyst],
            tasks=[clarify_the_ask.draft_process],
        )
        _ = crew.kickoff(
            inputs={
                "input_ask": input_ask,
            }
        )
        output = json.loads(clarify_the_ask.draft_file_json)
        return

    # @listen(fetch_leads)
    # def score_leads(self, leads):
    #     scores = lead_scoring_crew.kickoff_for_each(leads)
    #     self.state["score_crews_results"] = scores
    #     return scores

    # @listen(score_leads)
    # def store_leads_score(self, scores):
    #     # Here we would store the scores in the database
    #     return scores

    # @listen(score_leads)
    # def filter_leads(self, scores):
    #     return [score for score in scores if score["lead_score"].score > 70]

    # @listen(filter_leads)
    # def write_email(self, leads):
    #     scored_leads = [lead.to_dict() for lead in leads]
    #     emails = email_writing_crew.kickoff_for_each(scored_leads)
    #     return emails

    # @listen(write_email)
    # def send_email(self, emails):
    #     # Here we would send the emails to the leads
    #     return emails


flow = ProcessPipeline()
flow.plot()
