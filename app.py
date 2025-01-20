import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px
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
        r"(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),? (\d{1,2}:\d{2} ?(?:AM|PM|am|pm|\u202Fam|\u202Fpm)?) - (.*?): (.*)"
    )

    for line in data:
        match = pattern.match(line)
        if match:
            date, time, user, message = match.groups()
            messages.append((date, time, user, message))

    if not messages:
        st.error("No valid messages found. Please check the format.")
        st.write("First few lines of the file for debugging:")
        st.code('\n'.join(data[:10]))
        return pd.DataFrame()

    df = pd.DataFrame(messages, columns=["Date", "Time", "User", "Message"])
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"], format="%I:%M %p", errors="coerce").dt.time
    df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour
    df["Weekday"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month_name()
    return df


# Function to detect emojis using Unicode ranges
def detect_emojis(text):
    emoji_pattern = re.compile(
        u'['
        u'\U0001F600-\U0001F64F'  # Emoticons
        u'\U0001F300-\U0001F5FF'  # Symbols & Pictographs
        u'\U0001F680-\U0001F6FF'  # Transport & Map
        u'\U0001F700-\U0001F77F'  # Alchemical Symbols
        u'\U0001F780-\U0001F7FF'  # Geometric Shapes
        u'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
        u'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        u'\U0001FA00-\U0001FA6F'  # Chess Symbols
        u'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
        u'\U00002702-\U000027B0'  # Dingbats
        u'\U000024C2-\U0001F251'  # Enclosed Characters
        u'\U0001F004-\U0001F0CF'  # Playing Cards
        u'\U00002B50'  # Star Symbol
        u']', flags=re.UNICODE)

    return emoji_pattern.findall(text)


# Function for emoji analysis
def analyze_emojis(df):
    emojis_list = [c for message in df['Message'].dropna() for c in detect_emojis(message)]
    emoji_count = Counter(emojis_list)
    return emoji_count


# Function to analyze common words
def common_words(df):
    all_words = " ".join(df['Message'].dropna())
    words = re.findall(r'\w+', all_words.lower())
    word_count = Counter(words)
    common_words_df = pd.DataFrame(word_count.most_common(10), columns=["Word", "Frequency"])
    return common_words_df


# Function to perform sentiment analysis
def sentiment_analysis(df):
    sentiments = df['Message'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
    return sentiments


# Function to display chat sentiment trend over time
def sentiment_trend(df):
    df['Sentiment'] = df['Message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    sentiment_daily = df.groupby(df['Date'].dt.date)['Sentiment'].mean()
    fig_sentiment = px.line(sentiment_daily, x=sentiment_daily.index, y=sentiment_daily.values,
                            title="Chat Sentiment Trend Over Time", labels={"x": "Date", "y": "Sentiment Polarity"})
    st.plotly_chart(fig_sentiment)


# Function to display weekly activity (activity by day of the week)
def weekly_activity(df):
    weekly_activity = df.groupby('Weekday')['Message'].count().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig_weekly = px.bar(weekly_activity, x=weekly_activity.index, y=weekly_activity.values,
                        title="Weekly Activity", labels={"x": "Weekday", "y": "Messages Sent"})
    st.plotly_chart(fig_weekly)


# Function to display daily activity (messages by day)
def daily_activity(df):
    daily_activity = df.groupby(df['Date'].dt.date)['Message'].count()
    fig_daily = px.bar(daily_activity, x=daily_activity.index, y=daily_activity.values,
                       title="Daily Activity", labels={"x": "Date", "y": "Messages Sent"})
    st.plotly_chart(fig_daily)


# Function for activity maps (user-wise activity map)
def activity_maps(df):
    if df.empty:
        return

    busy_users = df['User'].value_counts(normalize=True) * 100
    busy_users_df = pd.DataFrame(busy_users).reset_index()
    busy_users_df.columns = ["User", "Percentage"]

    most_busy_user = busy_users_df.iloc[0] if not busy_users_df.empty else None
    if most_busy_user is not None:
        st.write(f"👑 **Most Active User in this Chat**: {most_busy_user['User']} ({most_busy_user['Percentage']:.2f}%)")

    fig_users = px.bar(busy_users_df, x="User", y="Percentage", title="Most Busy Users",
                       labels={"User": "User", "Percentage": "Percentage of Messages"})
    st.plotly_chart(fig_users)


# Function to display most busy day
def busy_day(df):
    busy_day = df.groupby(df['Date'].dt.day_name())['Message'].count()
    busy_day_sorted = busy_day.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig_busy_day = px.bar(busy_day_sorted, x=busy_day_sorted.index, y=busy_day_sorted.values,
                          title="Most Busy Day", labels={"x": "Day of the Week", "y": "Messages Sent"})
    st.plotly_chart(fig_busy_day)


# Function to display most busy month
def busy_month(df):
    busy_month = df.groupby(df['Date'].dt.month_name())['Message'].count()
    busy_month_sorted = busy_month.reindex(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
         "December"]
    )
    fig_busy_month = px.bar(busy_month_sorted, x=busy_month_sorted.index, y=busy_month_sorted.values,
                            title="Most Busy Month", labels={"x": "Month", "y": "Messages Sent"})
    st.plotly_chart(fig_busy_month)


# Function to show top message counts
def top_messages_count(df):
    message_counts = df['Message'].value_counts().head(10)
    fig_top_messages = px.bar(message_counts, x=message_counts.index, y=message_counts.values,
                              title="Top 10 Most Frequent Messages", labels={"x": "Message", "y": "Frequency"})
    st.plotly_chart(fig_top_messages)


# Function to analyze message length
def message_length_analysis(df):
    df['Message Length'] = df['Message'].apply(lambda x: len(str(x)))
    avg_length = df['Message Length'].mean()
    fig_message_length = px.histogram(df, x="Message Length", title="Message Length Distribution",
                                      labels={"Message Length": "Length of Messages"})
    st.plotly_chart(fig_message_length)
    st.write(f"Average Message Length: {avg_length:.2f} characters")


# Function to perform user-wise sentiment analysis
def user_sentiment_analysis(df):
    user_sentiment = df.groupby('User').apply(lambda x: TextBlob(" ".join(x['Message'])).sentiment.polarity)
    user_sentiment_df = user_sentiment.reset_index()
    user_sentiment_df.columns = ['User', 'Sentiment']
    fig_user_sentiment = px.bar(user_sentiment_df, x="User", y="Sentiment", title="User-wise Sentiment Analysis",
                                labels={"User": "User", "Sentiment": "Sentiment Polarity"})
    st.plotly_chart(fig_user_sentiment)


# Function to create a word cloud
def generate_word_cloud(df):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df['Message'].dropna()))
    st.image(wordcloud.to_array(), caption="Word Cloud of Chat Messages")


# Function to display most common words graph
def common_words_graph(df):
    common_word_df = common_words(df)
    fig_common_words = px.bar(common_word_df, x="Word", y="Frequency", title="Most Common Words Used",
                              labels={"Word": "Word", "Frequency": "Frequency"})
    st.plotly_chart(fig_common_words)


# Streamlit UI
st.title("📊 Ultimate WhatsApp Chat Analyzer")
uploaded_file = st.file_uploader("📂 Upload WhatsApp chat file", type=["txt"])

if uploaded_file is not None:
    df = preprocess_chat(uploaded_file)
    if not df.empty:
        total_messages = df.shape[0]
        total_words = df['Message'].apply(lambda x: len(str(x).split())).sum()
        media_shared = df['Message'].str.contains("<Media omitted>", na=False).sum()
        links_shared = df['Message'].str.contains(r"https?://", na=False).sum()

        # Display Chat Statistics
        st.subheader("📊 Chat Statistics")
        st.write(f"📝 Total Messages: {total_messages}")
        st.write(f"🔠 Total Words: {total_words}")
        st.write(f"📸 Media Shared: {media_shared}")
        st.write(f"🔗 Links Shared: {links_shared}")

        # Display sentiment trend
        sentiment_trend(df)

        # Display activity maps
        activity_maps(df)

        # Display top messages count
        top_messages_count(df)

        # Display message length distribution
        message_length_analysis(df)

        # Generate Word Cloud
        generate_word_cloud(df)

        # Display user sentiment analysis
        user_sentiment_analysis(df)

        # Display common words graph
        common_words_graph(df)
