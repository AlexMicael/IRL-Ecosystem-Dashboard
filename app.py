import pandas as pd
import altair as alt
import streamlit as st

from scipy import stats
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IRL Streaming Ecosystem Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING  ---
@st.cache_data
def load_data():
    """Loads and pre-processes all datasets with memory optimization."""
    try:
        # Twitch Stream Snapshots - only load essential columns
        twitch_df = pd.read_parquet("data/twitch_streams_data.parquet", 
                                     columns=['user_id', 'viewer_count', 'collection_timestamp', 
                                             'stream_title', 'started_at'])
        twitch_df['collection_timestamp'] = pd.to_datetime(twitch_df['collection_timestamp'], utc=True).dt.tz_localize(None)
        
        # Optimize data types
        twitch_df['viewer_count'] = pd.to_numeric(twitch_df['viewer_count'], downcast='integer')
        
        # Twitch Users
        users_df = pd.read_parquet("data/twitch_users_data.parquet")
        
        # YouTube Videos
        videos_df = pd.read_parquet("data/youtube_videos_data.parquet")
        videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], utc=True).dt.tz_localize(None)
        
        # Streamer Mapping
        map_df = pd.read_parquet("data/streamer_map.parquet")

        # YouTube Comments (Split into 2 Parts) - load in chunks
        comments_part1 = pd.read_parquet("data/youtube_comments_data_part1.parquet")
        comments_part2 = pd.read_parquet("data/youtube_comments_data_part2.parquet")
        comments_df = pd.concat([comments_part1, comments_part2], ignore_index=True)
        
        # Free up memory
        del comments_part1, comments_part2
        
        comments_df['published_at'] = pd.to_datetime(comments_df['published_at'], utc=True, errors='coerce').dt.tz_localize(None)
        
        # Optimize comment data types
        if 'toxicity_score' in comments_df.columns:
            comments_df['toxicity_score'] = pd.to_numeric(comments_df['toxicity_score'], downcast='float')
        if 'like_count' in comments_df.columns:
            comments_df['like_count'] = pd.to_numeric(comments_df['like_count'], downcast='integer')
        
        return twitch_df, users_df, videos_df, comments_df, map_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please ensure all parquet files are in the correct directory.")
        return None, None, None, None, None

# Load data once
twitch_df, users_df, videos_df, comments_df, map_df = load_data()

if twitch_df is None:
    st.stop()

# --- PRECOMPUTE AGGREGATIONS ---
@st.cache_data
def precompute_aggregations(_twitch_df, _videos_df, _comments_df):
    """Pre-aggregate heavy computations to reduce memory pressure."""
    
    # Daily aggregations
    twitch_daily = _twitch_df.set_index('collection_timestamp').resample('D').size().reset_index(name='count')
    twitch_daily['source'] = 'Twitch Snapshots'
    twitch_daily.rename(columns={'collection_timestamp': 'date'}, inplace=True)

    videos_daily = _videos_df.set_index('published_at').resample('D').size().reset_index(name='count')
    videos_daily['source'] = 'YouTube Videos'
    videos_daily.rename(columns={'published_at': 'date'}, inplace=True)
    
    comments_daily = _comments_df.set_index('published_at').resample('D').size().reset_index(name='count')
    comments_daily['source'] = 'YouTube Comments'
    comments_daily.rename(columns={'published_at': 'date'}, inplace=True)
    
    combined_daily = pd.concat([twitch_daily, videos_daily, comments_daily], ignore_index=True)
    
    # Twitch metrics per user
    twitch_metrics = _twitch_df.groupby('user_id').agg({
        'viewer_count': 'mean',
        'started_at': 'count'
    }).rename(columns={'viewer_count': 'avg_viewers', 'started_at': 'stream_count'}).reset_index()
    
    # Comments per video
    comments_per_video = _comments_df.groupby('video_id').size().reset_index(name='comment_count')
    
    return combined_daily, twitch_metrics, comments_per_video

combined_daily, twitch_metrics, comments_per_video = precompute_aggregations(twitch_df, videos_df, comments_df)

# --- HELPER FUNCTION: STATS FORMATTER ---
def display_stats(r, rho, slope, r2, p_val):
    """Helper to display regression stats nicely."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pearson (r)", f"{r:.4f}", help="Linear correlation (-1 to 1)")
    c2.metric("Spearman (rho)", f"{rho:.4f}", help="Rank correlation (-1 to 1)")
    c3.metric("Regression Slope", f"{slope:.4f}", help="Elasticity/Rate of change")
    c4.metric("R-Squared", f"{r2:.4f}", help="Variance explained (0 to 1)")
    
    if p_val < 0.001:
        p_text = "< 0.001 (Significant)"
        color = "green"
    else:
        p_text = f"{p_val:.4f} (Not Significant)"
        color = "red"
    st.caption(f"**Statistical Significance (P-value):** :{color}[{p_text}]")

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Home", "RQ1: Temporal Toxicity", "RQ2: Cross-Platform Predictor", "RQ3: Content Themes"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project 3: Interactive Dashboard**\n\n"
    "This tool analyzes the cross-platform ecosystem of IRL streamers on Twitch and YouTube."
)

# --- PAGE 1: DASHBOARD HOME ---
if page == "Dashboard Home":
    st.title("IRL Streaming Ecosystem Dashboard")
    st.markdown("""
    Welcome to the interactive data explorer for our CS 415 Project. This dashboard provides a live view into our collected dataset
    and allows you to explore the research questions defined in Project 2.
    """)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Twitch Snapshots", f"{len(twitch_df):,}")
    col2.metric("Unique Streamers", f"{len(users_df):,}")
    col3.metric("YouTube Videos", f"{len(videos_df):,}")
    col4.metric("YouTube Comments", f"{len(comments_df):,}")
    
    # Daily Collection Volume
    st.subheader("Daily Data Collection Volume")
    
    # Date Filter for Main Graph
    if not combined_daily.empty:
        combined_daily_copy = combined_daily.copy()
        combined_daily_copy['date'] = pd.to_datetime(combined_daily_copy['date'])
        combined_daily_copy['date_only'] = combined_daily_copy['date'].dt.date
        
        min_date = combined_daily_copy['date_only'].min()
        max_date = combined_daily_copy['date_only'].max()

        default_start_date = max_date - timedelta(days=90)
        if default_start_date < min_date:
            default_start_date = min_date
        
        date_range = st.date_input(
            "Filter Date Range",
            value=(default_start_date, max_date), 
            min_value=min_date,
            max_value=max_date,
            key="home_date_filter"
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (combined_daily_copy['date_only'] >= start_date) & (combined_daily_copy['date_only'] <= end_date)
            chart_data = combined_daily_copy.loc[mask]
        else:
            chart_data = combined_daily_copy
            
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='date:T',
            y='count:Q',
            color='source:N',
            tooltip=['date', 'count', 'source']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data available to display.")

    # Dataset Overview
    st.markdown("---")
    st.header("General Dataset Overview")
    st.markdown("Below are key findings from our exploratory analysis (Project 2), providing context for the specific RQs.")

    st.subheader("Toxicity Distribution (Histogram)")
    
    # Toxicity Histogram - sample for performance
    if not comments_df.empty and 'toxicity_score' in comments_df.columns:
        # Sample data if too large
        sample_size = min(50000, len(comments_df))
        toxicity_sample = comments_df['toxicity_score'].dropna().sample(n=sample_size, random_state=42)
        
        hist_data = pd.DataFrame({'toxicity_score': toxicity_sample})
        
        hist_chart = alt.Chart(hist_data).mark_bar().encode(
            x=alt.X("toxicity_score:Q", bin=alt.Bin(maxbins=50), title="Toxicity Score"),
            y=alt.Y('count()', title='Count'),
            tooltip=[alt.Tooltip('toxicity_score:Q', bin=True), alt.Tooltip('count()')]
        ).properties(title=f"Distribution of Comment Toxicity (Sample: {sample_size:,} comments)")
        
        st.altair_chart(hist_chart, use_container_width=True)
    else:
        st.warning("No comment data for toxicity histogram.")

    st.subheader("Cross-Platform Engagement (Scatter Plot)")
    if not twitch_df.empty and not comments_df.empty:
        video_counts = pd.merge(videos_df[['video_id', 'channel_id']], comments_per_video, on='video_id')
        avg_yt = video_counts.groupby('channel_id')['comment_count'].mean().reset_index()
        
        # Merge
        user_map_mini = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, left_on='login_name', right_on='twitch_login_name')
        
        merged_metrics = pd.merge(twitch_metrics, user_map_mini, on='user_id')
        merged_metrics = pd.merge(merged_metrics, avg_yt, left_on='youtube_channel_id', right_on='channel_id')
        
        if not merged_metrics.empty:
            scatter_chart = alt.Chart(merged_metrics).mark_circle(size=60).encode(
                x=alt.X('avg_viewers', scale=alt.Scale(type='log', nice=True), title='Avg Twitch Viewers (Log)'),
                y=alt.Y('comment_count', scale=alt.Scale(type='log', nice=True), title='Avg YouTube Comments (Log)'),
                tooltip=['display_name', 'avg_viewers', 'comment_count']
            ).properties(title="Twitch Viewership vs. YouTube Engagement").interactive()
            
            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.warning("Insufficient overlap data for scatter plot.")

    st.subheader("Top Content Keywords")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Top Twitch Keywords**")
        if not twitch_df.empty and 'stream_title' in twitch_df.columns:
            # Sample titles for performance
            sample_size = min(10000, len(twitch_df))
            title_sample = twitch_df['stream_title'].dropna().sample(n=sample_size, random_state=42)
            
            stop_words = "english"
            vec = CountVectorizer(stop_words=stop_words, max_features=10)
            try:
                bow = vec.fit_transform(title_sample.astype(str))
                word_counts = pd.DataFrame({'word': vec.get_feature_names_out(), 'count': bow.toarray().sum(axis=0)})
                word_counts = word_counts.sort_values('count', ascending=False)
                
                bar_twitch = alt.Chart(word_counts).mark_bar().encode(
                    x=alt.X('count', title='Frequency'),
                    y=alt.Y('word', sort='-x', title='Keyword'),
                    tooltip=['word', 'count']
                )
                st.altair_chart(bar_twitch, use_container_width=True)
            except ValueError:
                st.info("Not enough text data for Twitch analysis.")

    with col_b:
        st.markdown("**Top YouTube Keywords (Video Titles)**")
        if not videos_df.empty:
            vec_yt = CountVectorizer(stop_words="english", max_features=10)
            try:
                bow_yt = vec_yt.fit_transform(videos_df['video_title'].dropna().astype(str))
                word_counts_yt = pd.DataFrame({'word': vec_yt.get_feature_names_out(), 'count': bow_yt.toarray().sum(axis=0)})
                word_counts_yt = word_counts_yt.sort_values('count', ascending=False)
                
                bar_yt = alt.Chart(word_counts_yt).mark_bar(color='red').encode(
                    x=alt.X('count', title='Frequency'),
                    y=alt.Y('word', sort='-x', title='Keyword'),
                    tooltip=['word', 'count']
                )
                st.altair_chart(bar_yt, use_container_width=True)
            except ValueError:
                st.info("Not enough text data for YouTube analysis.")

# --- PAGE 2: RQ1 (TEMPORAL TOXICITY) ---
elif page == "RQ1: Temporal Toxicity":
    st.title("RQ1: Temporal Evolution of Toxicity")
    st.markdown("**Question:** How does toxicity evolve over time in response to creator-specific events?")

    merged_map = pd.merge(map_df, users_df, left_on='twitch_login_name', right_on='login_name')
    creator = st.sidebar.selectbox("Select Creator", merged_map['display_name'].unique())
    
    channel_id = merged_map[merged_map['display_name'] == creator]['youtube_channel_id'].values[0]
    creator_vids = videos_df[videos_df['channel_id'] == channel_id]['video_id']
    
    # Filter comments for this creator only
    df = comments_df[comments_df['video_id'].isin(creator_vids)].copy()
    
    if not df.empty:
        # Aggregation
        granularity = st.sidebar.select_slider("Granularity", ["D", "W", "M"], value="W")
        timeline = df.set_index('published_at').resample(granularity)['toxicity_score'].agg(['mean', 'count']).reset_index()
        timeline.rename(columns={'mean': 'avg_toxicity', 'count': 'comment_volume'}, inplace=True)

        # Statistics
        mean_tox = df['toxicity_score'].mean()
        max_tox = timeline['avg_toxicity'].max()
        std_tox = df['toxicity_score'].std()
        
        s1, s2, s3 = st.columns(3)
        s1.metric("Baseline Toxicity", f"{mean_tox:.3f}")
        s2.metric("Max Spike", f"{max_tox:.3f}", delta=f"{(max_tox-mean_tox):.3f} above avg", delta_color="inverse")
        s3.metric("Volatility (Std Dev)", f"{std_tox:.3f}")
        
        # Chart
        st.subheader(f"Toxicity Trends for {creator}")
        base = alt.Chart(timeline.reset_index()).encode(x='published_at:T')
        line = base.mark_line(color='red').encode(
            y=alt.Y('avg_toxicity', title='Avg Toxicity'),
            tooltip=['published_at', 'avg_toxicity']
        )
        bar = base.mark_bar(opacity=0.3).encode(
            y=alt.Y('comment_volume', title='Volume'),
            tooltip=['published_at', 'comment_volume']
        )
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent').interactive(), use_container_width=True)

        # Insights
        st.markdown("### Insights")
        st.write(f"Showing analysis for **{len(df)}** comments.")
        
        st.subheader("Most Toxic Periods")
        toxic_periods = timeline.sort_values('avg_toxicity', ascending=False).head(5)
        st.dataframe(toxic_periods)

# --- PAGE 3: RQ2 - CROSS-PLATFORM PREDICTOR ---
elif page == "RQ2: Cross-Platform Predictor":
    st.title("RQ2: Cross-Platform Engagement Prediction")
    st.markdown("**Question:** Do Twitch metrics (viewers, duration) predict YouTube outcomes?")
    
    st.sidebar.subheader("Regression Parameters")
    
    # Aggregate YouTube Metrics per Channel
    comments_with_channel = pd.merge(
        comments_df[['video_id', 'toxicity_score', 'like_count']], 
        videos_df[['video_id', 'channel_id']], 
        on='video_id'
    )
    
    yt_metrics = comments_with_channel.groupby('channel_id').agg({
        'video_id': 'count',
        'toxicity_score': 'mean',
        'like_count': 'mean'
    }).rename(columns={'video_id': 'total_comments', 'toxicity_score': 'avg_channel_toxicity', 'like_count': 'avg_comment_likes'}).reset_index()
    
    # Map Twitch User ID to YouTube Channel ID
    user_map = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, left_on='login_name', right_on='twitch_login_name')
    
    # Merge
    combined_metrics = pd.merge(twitch_metrics, user_map, on='user_id')
    combined_metrics = pd.merge(combined_metrics, yt_metrics, left_on='youtube_channel_id', right_on='channel_id')
    
    if combined_metrics.empty:
        st.error("Insufficient overlapping data between Twitch and YouTube to generate plot.")
    else:
        # Controls
        x_metric = st.sidebar.selectbox("X-Axis (Twitch)", ["avg_viewers", "stream_count"])
        y_metric = st.sidebar.selectbox("Y-Axis (YouTube)", ["total_comments", "avg_channel_toxicity", "avg_comment_likes"])
        
        valid_data = combined_metrics[[x_metric, y_metric]].dropna()
        
        if len(valid_data) < 3:
            st.error(f"Insufficient valid data for {x_metric} vs {y_metric}. Only {len(valid_data)} creators have both metrics.")
            st.info("This typically means toxicity scores are missing for most channels. Try selecting a different Y-axis metric.")
        else:
            valid_indices = valid_data.index
            plot_data = combined_metrics.loc[valid_indices]
            
            # Pearson
            pearson_r, p_p = stats.pearsonr(plot_data[x_metric], plot_data[y_metric])
            
            # Spearman
            spearman_r, s_p = stats.spearmanr(plot_data[x_metric], plot_data[y_metric])
            
            # Linear Regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[x_metric], plot_data[y_metric])
            r_squared = r_value**2
            
            display_stats(pearson_r, spearman_r, slope, r_squared, p_value)
            
            # Data Quality Info
            st.info(f"Analysis based on **{len(plot_data)}** creators with valid data for both metrics. "
                   f"({len(combined_metrics) - len(plot_data)} excluded due to missing toxicity scores)")

            # Plot
            st.subheader(f"{x_metric} vs. {y_metric}")
            
            scatter = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X(x_metric, scale=alt.Scale(zero=False)),
                y=alt.Y(y_metric, scale=alt.Scale(zero=False)),
                tooltip=['display_name', x_metric, y_metric]
            ).interactive()
            
            # Regression Line
            reg_line = scatter.transform_regression(x_metric, y_metric).mark_line(color='red')
            
            st.altair_chart(scatter + reg_line, use_container_width=True)
            
            # Data Table
            st.markdown("### Creator Data")
            st.dataframe(plot_data[['display_name', x_metric, y_metric]].sort_values(x_metric, ascending=False))

# --- PAGE 4: RQ3 (CONTENT THEMES) ---
elif page == "RQ3: Content Themes":
    st.title("RQ3: Content Theme Analyzer")
    st.markdown("**Question:** How do specific keywords influence engagement, and what related themes emerge?")
    
    keyword = st.sidebar.text_input("Keyword", "drama").lower()
    
    df = videos_df.copy()
    df['has_keyword'] = df['video_title'].str.lower().str.contains(keyword, na=False)
    
    # Merge Counts
    df = pd.merge(df, comments_per_video, on='video_id', how='left')
    df['comment_count'] = df['comment_count'].fillna(0)
    
    if df['has_keyword'].sum() > 0:
        group_yes = df[df['has_keyword']]['comment_count']
        group_no = df[~df['has_keyword']]['comment_count']
        
        st.markdown(f"### Statistical Test: Does '{keyword}' drive engagement?")
        
        # T-Test
        t_stat, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
        
        m1 = group_yes.mean()
        m2 = group_no.mean()
        lift = ((m1 - m2) / m2) * 100 if m2 > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Comments (With)", f"{m1:.1f}")
        c2.metric("Avg Comments (Without)", f"{m2:.1f}")
        c3.metric("Engagement Lift", f"{lift:+.1f}%", delta_color="normal")
        
        if p_val < 0.05:
            st.success(f"**Statistically Significant** (p = {p_val:.4e}). The keyword '{keyword}' has a real impact on engagement.")
        else:
            st.warning(f"**Not Significant** (p = {p_val:.4f}). The difference might be due to chance.")

        # Comparative Density Plot - sample for performance
        st.subheader("Engagement Distribution by Keyword Presence")
        
        # Sample data if too large
        sample_size = min(5000, len(df))
        chart_data = df[['has_keyword', 'comment_count']].sample(n=sample_size, random_state=42)
        chart_data['Type'] = chart_data['has_keyword'].map({True: 'With Keyword', False: 'Without Keyword'})
        
        chart = alt.Chart(chart_data).transform_density(
            'comment_count',
            as_=['comment_count', 'density'],
            groupby=['Type']
        ).mark_area(opacity=0.5).encode(
            x=alt.X('comment_count:Q', title='Comment Count'),
            y='density:Q',
            color='Type:N'
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Snowball Sampling
        st.subheader(f"Snowball Sampling: What co-occurs with '{keyword}'?")
        try:
            subset = df[df['has_keyword']]['video_title'].dropna()
            if len(subset) > 0:
                vec = CountVectorizer(stop_words='english', max_features=15)
                bow = vec.fit_transform(subset.astype(str))
                words = pd.DataFrame({'word': vec.get_feature_names_out(), 'count': bow.toarray().sum(axis=0)})
                words = words[words['word'] != keyword].sort_values('count', ascending=False)
                
                bar = alt.Chart(words).mark_bar().encode(
                    x='count:Q', y=alt.Y('word:N', sort='-x')
                )
                st.altair_chart(bar, use_container_width=True)
            else:
                st.info("Not enough data for snowball sampling.")
        except:
            st.info("Not enough data for snowball sampling.")

        # Show Examples
        st.subheader("Example Videos")
        st.dataframe(df[df['has_keyword']][['video_title', 'video_id']].head(10))
            
    else:
        st.error("Keyword not found in any titles.")