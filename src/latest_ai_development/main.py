from latest_ai_development.crew import LatestAiDevelopment
from datetime import datetime

def extract_topic_from_url(url):
    return url.rstrip('/').split('/')[-1].replace('-', ' ').replace('_', ' ').title()

def run():
    user_url = input("Enter a URL to analyze: ")
    topic = extract_topic_from_url(user_url)

    inputs = {
        'topic': topic,
        'user_url': user_url,
        'current_year': str(datetime.now().year)
    }

    # Initialize the crew instance
    crew_instance = LatestAiDevelopment()

    # ✅ No need to add to rag_tool here — that's done inside crew.py via config
    crew_instance.crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
