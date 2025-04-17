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

    try:
        # Run the crew with proper error handling
        crew_instance.crew().kickoff(inputs=inputs)
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check that Ollama is running at http://localhost:11434 and has the required models.")

if __name__ == "__main__":
    run()