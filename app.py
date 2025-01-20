import streamlit as st
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import datetime as dt



# Function to read and preprocess chat data
def preprocess_chat(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    data = content.splitlines()
    messages = []
    pattern = re.compile(
        r"(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),? (\d{1,2}:\d{2} ?(?:AM|PM|am|pm| am| pm)?) - (.*?): (.*)")

    for line in data:
        match = pattern.match(line)
        if match:
            date, time, user, message = match.groups()
            messages.append((date, time, user, message))

    if not messages:
        st.error("No valid messages found. Please check the format.")
        st.write("First few lines of the file for debugging:")
        st.code('\n'.join(data[:10]))

    df = pd.DataFrame(messages, columns=["Date", "Time", "User", "Message"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"] = df["Date"].dt.hour
    df["Weekday"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month_name()
    return df


# Function to analyze messages
def analyze_messages(df):
    total_messages = df.shape[0]
    total_words = df['Message'].apply(lambda x: len(str(x).split())).sum()
    media_shared = df['Message'].str.contains("<Media omitted>", na=False).sum()
    links_shared = df['Message'].str.contains(r"https?://", na=False).sum()
    return total_messages, total_words, media_shared, links_shared


# Function to generate word cloud
def generate_wordcloud(df):
    text = ' '.join(df['Message'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud


# Function for emoji analysis
def analyze_emojis(df):
    emojis = [c for message in df['Message'].dropna() for c in message if c in emoji.EMOJI_DATA]
    emoji_count = Counter(emojis)
    return emoji_count


# Function for sentiment analysis
def analyze_sentiment(df):
    sentiments = df['Message'].dropna().apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment'] = sentiments
    return df


# Function for timeline graphs
def plot_timeline(df):
    df["Month-Year"] = df["Date"].dt.strftime('%Y-%m')
    monthly_counts = df.groupby("Month-Year").size().reset_index(name='Message Count')
    daily_counts = df.groupby("Date").size().reset_index(name='Message Count')

    # Monthly Timeline (Bar Chart)
    fig1 = px.bar(monthly_counts, x="Month-Year", y="Message Count", title="Monthly Timeline",
                  labels={"Month-Year": "Month-Year", "Message Count": "Messages"})
    st.plotly_chart(fig1)

    # Daily Timeline (Bar Chart)
    fig2 = px.bar(daily_counts, x="Date", y="Message Count", title="Daily Timeline",
                  labels={"Date": "Date", "Message Count": "Messages"})
    st.plotly_chart(fig2)


# Function for activity maps
def activity_maps(df):
    busy_day = df['Weekday'].value_counts().idxmax()
    busy_month = df['Month'].value_counts().idxmax()
    busy_users = df['User'].value_counts(normalize=True) * 100

    # Most Busy Day (Bar Chart)
    fig_day = px.bar(df['Weekday'].value_counts(), title="Most Busy Day",
                     labels={"value": "Messages", "index": "Day of Week"})
    st.plotly_chart(fig_day)

    # Most Busy Month (Bar Chart)
    fig_month = px.bar(df['Month'].value_counts(), title="Most Busy Month",
                       labels={"value": "Messages", "index": "Month"})
    st.plotly_chart(fig_month)

    # Most Busy User (Bar Chart with Name and Percentage)
    busy_users_df = pd.DataFrame(busy_users).reset_index()
    busy_users_df.columns = ["User", "Percentage"]

    # Identify the most busy user
    most_busy_user = busy_users_df.iloc[busy_users_df["Percentage"].idxmax()]
    most_busy_user_name = most_busy_user["User"]
    most_busy_user_percentage = most_busy_user["Percentage"]

    st.write(f"👑 **Most Active User in this Chat**: {most_busy_user_name} ({most_busy_user_percentage:.2f}%)")

    # Bar chart for most busy users
    fig_users = px.bar(busy_users_df, x="User", y="Percentage", title="Most Busy Users",
                       labels={"User": "User", "Percentage": "Percentage of Messages"})
    st.plotly_chart(fig_users)


# Function for weekly activity map
def weekly_activity_map(df):
    df["Week"] = df["Date"].dt.isocalendar().week
    weekly_counts = df.groupby("Week").size().reset_index(name='Message Count')
    fig_week = px.bar(weekly_counts, x="Week", y="Message Count", title="Weekly Activity Map",
                      labels={"Week": "Week Number", "Message Count": "Messages"})
    st.plotly_chart(fig_week)


# Function for most common words
def most_common_words(df):
    words = ' '.join(df['Message'].dropna()).split()
    common_words = Counter(words).most_common(20)
    word_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
    fig = px.bar(word_df, x='Word', y='Count', title='Most Common Words', labels={"Word": "Word", "Count": "Frequency"})
    st.plotly_chart(fig)


# Function for top messages count
def top_messages(df):
    top_messages = df['Message'].value_counts().head(10)
    top_messages_df = pd.DataFrame(top_messages).reset_index()
    top_messages_df.columns = ['Message', 'Count']
    fig = px.bar(top_messages_df, x='Message', y='Count', title='Top Messages Count',
                 labels={"Message": "Message", "Count": "Count"})
    st.plotly_chart(fig)


# Streamlit UI
st.title("📊 Ultimate WhatsApp Chat Analyzer")
uploaded_file = st.file_uploader("📂 Upload WhatsApp chat file", type=["txt"])

if uploaded_file is not None:
    df = preprocess_chat(uploaded_file)
    if not df.empty:
        total_messages, total_words, media_shared, links_shared = analyze_messages(df)

        st.subheader("📊 Chat Statistics")
        st.write(f"📝 Total Messages: {total_messages}")
        st.write(f"🔠 Total Words: {total_words}")
        st.write(f"🖼️ Media Shared: {media_shared}")
        st.write(f"🔗 Links Shared: {links_shared}")

        # Generate WordCloud
        st.subheader("☁️ Word Cloud")
        wordcloud = generate_wordcloud(df)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Display various analyses
        st.subheader("🔥 Analysis Features")
        plot_timeline(df)
        activity_maps(df)
        weekly_activity_map(df)
        most_common_words(df)
        top_messages(df)

        # Sentiment Analysis
        df = analyze_sentiment(df)
        sentiment_counts = df["Sentiment"].apply(
            lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral").value_counts()
        fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                               title="📈 Chat Sentiment Trend")
        st.plotly_chart(fig_sentiment)

        # Emoji Analysis
        st.subheader("😊 Emoji Analysis")
        emoji_count = analyze_emojis(df)
        emoji_df = pd.DataFrame(emoji_count.items(), columns=["Emoji", "Count"])
        fig_emoji = px.bar(emoji_df, x="Emoji", y="Count", title="Most Used Emojis")
        st.plotly_chart(fig_emoji)
    else:
        st.write("⚠️ No valid messages found in the uploaded file. Please check the format.")

st.write("📤 Upload a WhatsApp chat file to get started!")


