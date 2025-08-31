import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import requests
from googleapiclient.discovery import build # python.exe -m pip install google-api-python-client

# 1. Load CSV
def load_csv(path):
    return pd.read_csv(path)

# 2. Load JSON
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.json_normalize(data)

# 3. Load TXT (unstructured, space-separated)
def load_txt(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["ID", "Name", "Role"])

# 4. Load XLXS
def load_xlsx(path, sheet_name=0):
    return pd.read_excel(path, sheet_name=sheet_name)

# 5. Load API
def load_api_json(url: str, timeout: int = 10) -> pd.DataFrame:
    """
    Collect JSON from an API and convert it into a DataFrame
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()  # Stops if API returns an error
    data = resp.json()

    # If there is a list of objects (normal structure)
    if isinstance(data, list):
        return pd.DataFrame(data)
    
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return pd.DataFrame(v)
        return pd.json_normalize(data)  # fallback

    return pd.DataFrame()  # Empty if nothing is recognized

# 6. Load Youtube
def load_youtube_metadata_api(video_id, api_key):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            print(f"No video found for ID: {video_id}")
            return pd.DataFrame()

        item = response["items"][0]
        data = {
            "title": item["snippet"]["title"],
            "author": item["snippet"]["channelTitle"],
            "publish_date": item["snippet"]["publishedAt"],
            "views": int(item["statistics"].get("viewCount", 0)),
            "likes": int(item["statistics"].get("likeCount", 0)),
            "comments": int(item["statistics"].get("commentCount", 0)),
            "description": item["snippet"]["description"],
            "duration_sec": parse_iso8601_duration(item["contentDetails"]["duration"])
        }
        return pd.DataFrame([data])

    except Exception as e:
        print(f"Failed to fetch YouTube metadata: {e}")
        return pd.DataFrame()


# Helper: convert ISO 8601 duration (e.g., PT4M13S) to seconds
def parse_iso8601_duration(duration):
    import re
    match = re.match(
        r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration
    )
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    return int(hours or 0)*3600 + int(minutes or 0)*60 + int(seconds or 0)


# Example: get data from a API 
if __name__ == "__main__":
    api_url = "https://jsonplaceholder.typicode.com/posts"
    api_data = load_api_json(api_url)

    print("API Data (første 5 rækker):")
    print(api_data.head())

# Example ingestion
csv_data = load_csv("C:/Users/Asim/Downloads/MP1/sample.csv") # Remember to change the path
json_data = load_json("C:/Users/Asim/Downloads/MP1/sample.json") # Remember to change the path
txt_data = load_txt("C:/Users/Asim/Downloads/MP1/sample.txt") # Remember to change the path
xlsx_data = load_xlsx("C:/Users/Asim/Downloads/MP1/sample.xlsx") # Remember to change the path

# Example YouTube video
API_KEY = # <-- Replace with your YouTube Data API key
video_id = "dQw4w9WgXcQ" # just the video ID, not full URL
yt_metadata = load_youtube_metadata_api(video_id, API_KEY)

print("YouTube Metadata:")
print(yt_metadata.head())
print("-" * 50)


# 6. Explore & clean
csv_data.info()               # don't wrap in print()
print(csv_data.head())        # optional: see first rows

csv_data.dropna(inplace=True)  # remove missing rows

xlsx_data.info()

# 7. Anonymisation (replace names with ID numbers)
if 'name' in csv_data.columns:
    csv_data = csv_data.copy()  # safe copy before assignment
    csv_data['name'] = [f"user_{i}" for i in range(len(csv_data))]

# 8. Visualisation

plt.figure(figsize=(10,5))
plt.hist(csv_data["age"], bins=5, rwidth=0.9)
plt.title("Age distribution (CSV)")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.show()

# Group and calculate mean price
avg_price = json_data.groupby("category")["price"].mean()

# Plot as bar chart
plt.figure(figsize=(8,5))
avg_price.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Average Price by Category (JSON)")
plt.xlabel("Category")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.75)
plt.tight_layout()
plt.show()

# ------------------------------
# YouTube visualization
# ------------------------------

if not yt_metadata.empty and {"title", "views"}.issubset(yt_metadata.columns):
    plt.figure(figsize=(6,4))
    plt.bar(yt_metadata["title"], yt_metadata["views"], color="orange", edgecolor="black")
    plt.title("YouTube Video Views")
    plt.ylabel("Views")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No YouTube metadata available to plot.")

# Example scatter
if {'age', 'salary'}.issubset(csv_data.columns):
    csv_data.plot.scatter(x='age', y='salary', title='Age vs Salary')
    plt.tight_layout()
    plt.show()


if {'age', 'salary'}.issubset(csv_data.columns):
    plt.figure(figsize=(8,6))
    plt.scatter(csv_data["age"], csv_data["salary"], 
                color="royalblue", edgecolor="black", s=100)

    plt.title("Age vs Salary (CSV)", fontsize=14)
    plt.xlabel("Age")
    plt.ylabel("Salary")
    plt.grid(True, linestyle="--", alpha=0.7)


# Optionally add labels for each point (if 'name' exists)
    if "name" in csv_data.columns:
        for i, txt in enumerate(csv_data["name"]):
            plt.annotate(txt, (csv_data["age"].iloc[i], csv_data["salary"].iloc[i]),
                         textcoords="offset points", xytext=(5,5), fontsize=9)

    plt.tight_layout()
    plt.show()
