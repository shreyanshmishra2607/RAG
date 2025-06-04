from serpapi import GoogleSearch

params = {
    "q": "what is python",
    "api_key": "9b845b80c4c9ebb05f8408a4c5d74eaf636965cd86ecbc2d4a7421186dd07aa2",
    "engine": "google",
    "num": 5
}

search = GoogleSearch(params)
results = search.get_dict()

print(results)