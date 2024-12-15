# Social listening on Reddit using Python
# By: Joe Domaleski
# See blog post on https://blog.marketingdatascience.ai for details
# December 2024

import requests  # For making HTTP requests to Reddit's API
import csv  # For saving fetched comments into CSV format
import nltk  # For natural language processing
from nltk.corpus import stopwords  # For filtering out common stop words
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # For sentiment analysis
from collections import Counter, defaultdict  # For counting occurrences and grouping data
from wordcloud import WordCloud  # For creating word cloud visualizations
import matplotlib.pyplot as plt  # For creating visualizations

# Ensure necessary NLTK data is downloaded
nltk.download("vader_lexicon")  # Download VADER lexicon for sentiment analysis
nltk.download("stopwords")  # Download stop words for filtering text

# Global Variables and Constants
# Change these to your Reddit settings
CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
USERNAME = 'your-reddit-username'
PASSWORD = 'your-reddit-password'
USER_AGENT = "desktop:social-listener:v1.0 (by u/your-username)"
BRANDS = ["hunter", "hampton bay", "westinghouse", "bunnings", "big ass fans", "minka"]  # List of relevant brands

# Authenticate with Reddit API
def get_oauth_token():
    client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    post_data = {"grant_type": "password", "username": USERNAME, "password": PASSWORD}
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers
        )
        if response.status_code == 200:
            print("Authentication succeeded!")
            return response.json()["access_token"]
        else:
            print("Authentication failed!")
            print(response.json())
            return None
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None

# Fetch comments and subreddit information
def fetch_ceiling_fan_comments(access_token, search_query="ceiling fans", submission_limit=10):
    headers = {"Authorization": f"bearer {access_token}", "User-Agent": USER_AGENT}
    submissions_url = f"https://oauth.reddit.com/search?q={search_query}&limit={submission_limit}&sort=relevance&type=link"
    submissions_response = requests.get(submissions_url, headers=headers)
    if submissions_response.status_code != 200:
        print("Failed to fetch submissions.")
        return None, None

    submissions = submissions_response.json()["data"]["children"]
    all_comments = []
    subreddit_counts = Counter()

    for submission in submissions:
        subreddit = submission["data"]["subreddit"]
        subreddit_counts[subreddit] += 1
        submission_id = submission["data"]["id"]

        comments_url = f"https://oauth.reddit.com/comments/{submission_id}?limit=100"
        comments_response = requests.get(comments_url, headers=headers)
        if comments_response.status_code == 200:
            comments = comments_response.json()[1]["data"]["children"]
            for comment in comments:
                comment["subreddit"] = subreddit  # Attach subreddit to comment data
            all_comments.extend(comments)

    # Recalculate subreddit counts based on comment-level data
    for comment in all_comments:
        if "subreddit" in comment:
            subreddit_counts[comment["subreddit"]] += 1

    return all_comments, subreddit_counts

# Save fetched comments to a CSV file
def save_comments_to_csv(comments, filename="fetched_comments.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Author", "Comment Body", "Subreddit", "Permalink"])
        for comment in comments:
            if "body" in comment["data"]:
                writer.writerow([
                    comment["data"].get("author", "N/A"),
                    comment["data"].get("body", "N/A"),
                    comment.get("subreddit", "N/A"),
                    f"https://reddit.com{comment['data'].get('permalink', '')}"
                ])
    print(f"Fetched comments saved to {filename}")

# Analyze comments for sentiment, topics, and brands
def analyze_comments(comments):
    stop_words = set(stopwords.words("english"))
    vader = SentimentIntensityAnalyzer()

    sentiment_data = {"positive": 0, "neutral": 0, "negative": 0}
    topic_counts = Counter()
    brand_counts = Counter()
    topic_comments = defaultdict(list)  # Store comments grouped by topics
    all_words = []
    positive_words = []
    negative_words = []
    scored_comments = []

    topics = [
        "price", "installation", "wiring", "brands", "likes", "dislikes",
        "quality", "design", "noise", "blades", "remote", "cost", "light",
        "cheap", "expensive", "efficiency", "maintenance", "airflow"
    ]

    for comment in comments:
        if "body" in comment["data"]:
            body = comment["data"]["body"]
            permalink = f"https://reddit.com{comment['data']['permalink']}"
            sentiment = vader.polarity_scores(body)["compound"]
            scored_comments.append((sentiment, body, permalink))

            if sentiment > 0.05:
                sentiment_data["positive"] += 1
                positive_words.extend(body.split())
            elif sentiment < -0.05:
                sentiment_data["negative"] += 1
                negative_words.extend(body.split())
            else:
                sentiment_data["neutral"] += 1

            words = [word.lower() for word in body.split() if word.isalpha() and word.lower() not in stop_words]
            all_words.extend(words)

            for topic in topics:
                if topic in body.lower():
                    topic_counts[topic] += 1
                    topic_comments[topic].append(body)

            for brand in BRANDS:
                if brand in body.lower():
                    brand_counts[brand] += 1

    # Sort comments by sentiment for positive and negative lists
    scored_comments.sort(key=lambda x: x[0], reverse=True)
    most_positive = scored_comments[:10]
    most_negative = scored_comments[-10:]

    return sentiment_data, topic_counts, brand_counts, topic_comments, all_words, positive_words, negative_words, most_positive, most_negative

# Print comments grouped by topic
def print_comments_grouped_by_topic(topic_comments):
    print("\n--- Comments Grouped by Topic ---")
    for topic, comments in topic_comments.items():
        print(f"\n{topic.capitalize()} ({len(comments)} mentions):")
        for comment in comments[:5]:  # Display up to 5 comments per topic
            print(f"- {comment}")

# Visualize results
def visualize_results(sentiment_data, subreddit_counts, all_words, positive_words, negative_words, brand_counts):
    # Sentiment pie chart
    plt.figure()
    plt.pie(
        sentiment_data.values(),
        labels=sentiment_data.keys(),
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Sentiment Distribution")
    plt.show()

    # Word cloud for all comments
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Common Words")
    plt.show()

    # Combine positive and negative word clouds in one figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns
    positive_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(positive_words))
    negative_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(negative_words))

    # Positive Word Cloud
    axes[0].imshow(positive_wordcloud, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Positive Word Cloud", fontsize=16)

    # Negative Word Cloud
    axes[1].imshow(negative_wordcloud, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Negative Word Cloud", fontsize=16)

    plt.tight_layout()  # Ensure no overlapping text or titles
    plt.show()

    # Top 20 Word Frequency Bar Chart
    word_freq = Counter(all_words).most_common(20)
    words, counts = zip(*word_freq)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title("Top 20 Word Frequency")
    plt.show()

    # Subreddit Activity Bar Chart
    sorted_subreddits = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
    subreddits, counts = zip(*sorted_subreddits)
    plt.figure(figsize=(12, 6))
    plt.bar(subreddits, counts)
    plt.xticks(rotation=45, ha="right")  # Rotate and align text to prevent cutoff
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.title("Subreddit Activity")
    plt.show()

    # Brand Mentions Bar Chart
    sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
    brands, brand_counts_values = zip(*sorted_brands)
    plt.figure(figsize=(12, 6))
    plt.bar(brands, brand_counts_values)
    plt.xticks(rotation=45, ha="right")  # Rotate and align text to prevent cutoff
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.title("Brand Mentions")
    plt.show()

# Summarize key findings
def generate_final_summary(comments, subreddit_counts, brand_counts, topic_counts, most_positive, most_negative):
    user_activity = Counter(comment["data"]["author"] for comment in comments if "author" in comment["data"])
    top_users = user_activity.most_common(5)

    print("\n--- Final Summary ---")
    print(f"Total Comments Fetched: {len(comments)}")
    print(f"Total Subreddits Involved: {len(subreddit_counts)}")
    print("Most Mentioned Brands:")
    for brand, count in brand_counts.most_common(5):
        print(f"- {brand}: {count} mentions")
    print("Most Frequent Topics:")
    for topic, count in topic_counts.most_common(5):
        print(f"- {topic}: {count} mentions")
    print("Top Users (Potential Influencers):")
    for user, count in top_users:
        print(f"- {user}: {count} comments")

    print("\nTop Positive Comment Example:")
    sentiment, comment, link = most_positive[0]
    print(f"Sentiment Score: {sentiment}\nComment: {comment}\nLink: {link}")

    print("\nTop Negative Comment Example:")
    sentiment, comment, link = most_negative[0]
    print(f"Sentiment Score: {sentiment}\nComment: {comment}\nLink: {link}")

# Main Execution
if __name__ == "__main__":
    token = get_oauth_token()
    if token:
        comments, subreddit_counts = fetch_ceiling_fan_comments(token)
        if comments:
            print(f"Number of Comments Fetched: {len(comments)}")
            
            # Save comments to CSV
            save_comments_to_csv(comments)
            
            sentiment_data, topic_counts, brand_counts, topic_comments, all_words, positive_words, negative_words, most_positive, most_negative = analyze_comments(comments)
            print_comments_grouped_by_topic(topic_comments)
            visualize_results(sentiment_data, subreddit_counts, all_words, positive_words, negative_words, brand_counts)
            generate_final_summary(comments, subreddit_counts, brand_counts, topic_counts, most_positive, most_negative)
