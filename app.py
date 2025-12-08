import pandas as pd
import altair as alt
import streamlit as st
import gc

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

# --- LAZY DATA LOADING  ---
@st.cache_data(ttl=3600)
def load_twitch_data():
    """Load only Twitch data."""
    try:
        df = pd.read_parquet("data/twitch_streams_data.parquet", 
                            columns=['user_id', 'viewer_count', 'collection_timestamp', 
                                    'stream_title', 'started_at'])
        df['collection_timestamp'] = pd.to_datetime(df['collection_timestamp'], utc=True).dt.tz_localize(None)
        df['viewer_count'] = pd.to_numeric(df['viewer_count'], downcast='integer')
        return df
    except Exception as e:
        st.error(f"Error loading Twitch data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_users_data():
    """Load users data."""
    try:
        return pd.read_parquet("data/twitch_users_data.parquet")
    except Exception as e:
        st.error(f"Error loading users data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_videos_data():
    """Load YouTube videos data."""
    try:
        df = pd.read_parquet("data/youtube_videos_data.parquet")
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True).dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error loading videos data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_map_data():
    """Load streamer mapping data."""
    try:
        return pd.read_parquet("data/streamer_map.parquet")
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, max_entries=1)
def load_comments_lightweight():
    """Load comments with only essential columns and aggressive optimization."""
    try:
        # Load minimal columns
        cols = ['video_id', 'published_at', 'toxicity_score', 'like_count']
        
        # Try loading part 1
        try:
            df1 = pd.read_parquet("data/youtube_comments_data_part1.parquet", columns=cols)
        except:
            df1 = pd.read_parquet("data/youtube_comments_data_part1.parquet")
            df1 = df1[cols] if all(c in df1.columns for c in cols) else df1
        
        # Try loading part 2
        try:
            df2 = pd.read_parquet("data/youtube_comments_data_part2.parquet", columns=cols)
        except:
            df2 = pd.read_parquet("data/youtube_comments_data_part2.parquet")
            df2 = df2[cols] if all(c in df2.columns for c in cols) else df2
        
        # Concatenate
        df = pd.concat([df1, df2], ignore_index=True)
        del df1, df2
        gc.collect()
        
        # Optimize data types
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce').dt.tz_localize(None)
        
        if 'toxicity_score' in df.columns:
            df['toxicity_score'] = pd.to_numeric(df['toxicity_score'], downcast='float', errors='coerce')
        if 'like_count' in df.columns:
            df['like_count'] = pd.to_numeric(df['like_count'], downcast='integer', errors='coerce')
        
        # Convert video_id to category to save memory
        df['video_id'] = df['video_id'].astype('category')
        
        return df
    except Exception as e:
        st.error(f"Error loading comments data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_comment_counts():
    """Pre-aggregate comment counts to avoid loading full dataset."""
    comments = load_comments_lightweight()
    if comments.empty:
        return pd.DataFrame()
    return comments.groupby('video_id', observed=True).size().reset_index(name='comment_count')

@st.cache_data(ttl=3600)
def get_daily_stats():
    """Pre-compute daily statistics."""
    twitch_df = load_twitch_data()
    videos_df = load_videos_data()
    comments_df = load_comments_lightweight()
    
    results = []
    
    if not twitch_df.empty:
        twitch_daily = twitch_df.set_index('collection_timestamp').resample('D').size().reset_index(name='count')
        twitch_daily['source'] = 'Twitch Snapshots'
        twitch_daily.rename(columns={'collection_timestamp': 'date'}, inplace=True)
        results.append(twitch_daily)
    
    if not videos_df.empty:
        videos_daily = videos_df.set_index('published_at').resample('D').size().reset_index(name='count')
        videos_daily['source'] = 'YouTube Videos'
        videos_daily.rename(columns={'published_at': 'date'}, inplace=True)
        results.append(videos_daily)
    
    if not comments_df.empty:
        comments_daily = comments_df.set_index('published_at').resample('D').size().reset_index(name='count')
        comments_daily['source'] = 'YouTube Comments'
        comments_daily.rename(columns={'published_at': 'date'}, inplace=True)
        results.append(comments_daily)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

@st.cache_data(ttl=3600)
def get_twitch_metrics():
    """Pre-compute Twitch metrics per user."""
    twitch_df = load_twitch_data()
    if twitch_df.empty:
        return pd.DataFrame()
    
    return twitch_df.groupby('user_id').agg({
        'viewer_count': 'mean',
        'started_at': 'count'
    }).rename(columns={'viewer_count': 'avg_viewers', 'started_at': 'stream_count'}).reset_index()

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
    **Authors:** Alex Chen Hsieh & Derek Li
                
    Welcome to the interactive data explorer for our CS 415 Project. This dashboard provides a live view into our collected dataset
    and allows you to explore the research questions defined in Project 2.
    """)
    
    # Load data progressively
    with st.spinner("Loading data..."):
        twitch_df = load_twitch_data()
        users_df = load_users_data()
        videos_df = load_videos_data()
        comments_df = load_comments_lightweight()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Twitch Snapshots", f"{len(twitch_df):,}")
    col2.metric("Unique Streamers", f"{len(users_df):,}")
    col3.metric("YouTube Videos", f"{len(videos_df):,}")
    col4.metric("YouTube Comments", f"{len(comments_df):,}")
    
    # Daily Collection Volume
    st.subheader("Daily Data Collection Volume")
    
    combined_daily = get_daily_stats()
    
    if not combined_daily.empty:
        combined_daily['date'] = pd.to_datetime(combined_daily['date'])
        combined_daily['date_only'] = combined_daily['date'].dt.date
        
        min_date = combined_daily['date_only'].min()
        max_date = combined_daily['date_only'].max()

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
            mask = (combined_daily['date_only'] >= start_date) & (combined_daily['date_only'] <= end_date)
            chart_data = combined_daily.loc[mask]
        else:
            chart_data = combined_daily
            
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='date:T',
            y='count:Q',
            color='source:N',
            tooltip=['date', 'count', 'source']
        ).interactive()
        
        st.altair_chart(chart, width='stretch')
    else:
        st.info("No data available to display.")

    # Dataset Overview
    st.markdown("---")
    st.header("General Dataset Overview")

    st.subheader("Toxicity Distribution (Histogram)")
    
    if not comments_df.empty and 'toxicity_score' in comments_df.columns:
        # Heavy sampling for histogram
        sample_size = min(10000, len(comments_df))
        toxicity_sample = comments_df['toxicity_score'].dropna().sample(n=sample_size, random_state=42)
        
        hist_chart = alt.Chart(pd.DataFrame({'toxicity_score': toxicity_sample})).mark_bar().encode(
            x=alt.X("toxicity_score:Q", bin=alt.Bin(maxbins=50), title="Toxicity Score"),
            y=alt.Y('count()', title='Count')
        ).properties(title=f"Distribution of Comment Toxicity (Sample: {sample_size:,} comments)")
        
        st.altair_chart(hist_chart, width='stretch')
    else:
        st.warning("No comment data for toxicity histogram.")

    st.subheader("Cross-Platform Engagement (Scatter Plot)")
    if not twitch_df.empty and not comments_df.empty:
        comment_counts = get_comment_counts()
        video_counts = pd.merge(videos_df[['video_id', 'channel_id']], comment_counts, on='video_id')
        avg_yt = video_counts.groupby('channel_id')['comment_count'].mean().reset_index()
        
        twitch_metrics = get_twitch_metrics()
        map_df = load_map_data()
        user_map_mini = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, 
                                 left_on='login_name', right_on='twitch_login_name')
        
        merged_metrics = pd.merge(twitch_metrics, user_map_mini, on='user_id')
        merged_metrics = pd.merge(merged_metrics, avg_yt, left_on='youtube_channel_id', right_on='channel_id')
        
        if not merged_metrics.empty:
            scatter_chart = alt.Chart(merged_metrics).mark_circle(size=60).encode(
                x=alt.X('avg_viewers', scale=alt.Scale(type='log', nice=True), title='Avg Twitch Viewers (Log)'),
                y=alt.Y('comment_count', scale=alt.Scale(type='log', nice=True), title='Avg YouTube Comments (Log)'),
                tooltip=['display_name', 'avg_viewers', 'comment_count']
            ).properties(title="Twitch Viewership vs. YouTube Engagement").interactive()
            
            st.altair_chart(scatter_chart, width='stretch')

    st.subheader("Top Content Keywords")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Top Twitch Keywords**")
        if not twitch_df.empty and 'stream_title' in twitch_df.columns:
            sample_size = min(5000, len(twitch_df))
            title_sample = twitch_df['stream_title'].dropna().sample(n=sample_size, random_state=42)
            
            vec = CountVectorizer(stop_words="english", max_features=10)
            try:
                bow = vec.fit_transform(title_sample.astype(str))
                word_counts = pd.DataFrame({'word': vec.get_feature_names_out(), 'count': bow.toarray().sum(axis=0)})
                word_counts = word_counts.sort_values('count', ascending=False)
                
                bar_twitch = alt.Chart(word_counts).mark_bar().encode(
                    x=alt.X('count', title='Frequency'),
                    y=alt.Y('word', sort='-x', title='Keyword')
                )
                st.altair_chart(bar_twitch, width='stretch')
            except:
                st.info("Not enough text data for analysis.")

    with col_b:
        st.markdown("**Top YouTube Keywords**")
        if not videos_df.empty:
            vec_yt = CountVectorizer(stop_words="english", max_features=10)
            try:
                bow_yt = vec_yt.fit_transform(videos_df['video_title'].dropna().astype(str))
                word_counts_yt = pd.DataFrame({'word': vec_yt.get_feature_names_out(), 'count': bow_yt.toarray().sum(axis=0)})
                word_counts_yt = word_counts_yt.sort_values('count', ascending=False)
                
                bar_yt = alt.Chart(word_counts_yt).mark_bar(color='red').encode(
                    x=alt.X('count', title='Frequency'),
                    y=alt.Y('word', sort='-x', title='Keyword')
                )
                st.altair_chart(bar_yt, width='stretch')
            except:
                st.info("Not enough text data for analysis.")

# --- PAGE 2: RQ1 (TEMPORAL TOXICITY) ---
elif page == "RQ1: Temporal Toxicity":
    st.title("RQ1: Temporal Evolution of Toxicity")
    st.markdown("**Question:** How does toxicity evolve over time in response to creator-specific events?")

    with st.spinner("Loading creator data..."):
        users_df = load_users_data()
        map_df = load_map_data()
        videos_df = load_videos_data()
        comments_df = load_comments_lightweight()
    
    merged_map = pd.merge(map_df, users_df, left_on='twitch_login_name', right_on='login_name')
    creator = st.sidebar.selectbox("Select Creator", merged_map['display_name'].unique())
    
    channel_id = merged_map[merged_map['display_name'] == creator]['youtube_channel_id'].values[0]
    creator_vids = videos_df[videos_df['channel_id'] == channel_id]['video_id'].unique()
    
    # Filter early
    df = comments_df[comments_df['video_id'].isin(creator_vids)].copy()
    
    if not df.empty and 'toxicity_score' in df.columns:
        granularity = st.sidebar.select_slider("Granularity", ["D", "W", "M"], value="W")
        timeline = df.set_index('published_at').resample(granularity)['toxicity_score'].agg(['mean', 'count']).reset_index()
        timeline.rename(columns={'mean': 'avg_toxicity', 'count': 'comment_volume'}, inplace=True)

        mean_tox = df['toxicity_score'].mean()
        max_tox = timeline['avg_toxicity'].max()
        std_tox = df['toxicity_score'].std()
        
        s1, s2, s3 = st.columns(3)
        s1.metric("Baseline Toxicity", f"{mean_tox:.3f}")
        s2.metric("Max Spike", f"{max_tox:.3f}", delta=f"{(max_tox-mean_tox):.3f} above avg", delta_color="inverse")
        s3.metric("Volatility (Std Dev)", f"{std_tox:.3f}")
        
        st.subheader(f"Toxicity Trends for {creator}")
        base = alt.Chart(timeline).encode(x='published_at:T')
        line = base.mark_line(color='red').encode(y=alt.Y('avg_toxicity', title='Avg Toxicity'))
        bar = base.mark_bar(opacity=0.3).encode(y=alt.Y('comment_volume', title='Volume'))
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent').interactive(), width='stretch')

        st.markdown("### Insights")
        st.write(f"Showing analysis for **{len(df)}** comments.")
        
        st.subheader("Most Toxic Periods")
        st.dataframe(timeline.sort_values('avg_toxicity', ascending=False).head(5))
    else:
        st.warning("No toxicity data available for this creator.")

# --- PAGE 3: RQ2 - CROSS-PLATFORM PREDICTOR ---
elif page == "RQ2: Cross-Platform Predictor":
    st.title("RQ2: Cross-Platform Engagement Prediction")
    st.markdown("**Question:** Do Twitch metrics (viewers, duration) predict YouTube outcomes?")
    
    st.sidebar.subheader("Regression Parameters")
    
    with st.spinner("Computing cross-platform metrics..."):
        twitch_metrics = get_twitch_metrics()
        users_df = load_users_data()
        map_df = load_map_data()
        videos_df = load_videos_data()
        comments_df = load_comments_lightweight()
    
    if comments_df.empty:
        st.error("No comment data available.")
    else:
        # Aggregate YouTube metrics
        comments_subset = comments_df[['video_id', 'toxicity_score', 'like_count']].copy()
        comments_with_channel = pd.merge(comments_subset, videos_df[['video_id', 'channel_id']], on='video_id')
        
        yt_metrics = comments_with_channel.groupby('channel_id').agg({
            'video_id': 'count',
            'toxicity_score': 'mean',
            'like_count': 'mean'
        }).rename(columns={'video_id': 'total_comments', 'toxicity_score': 'avg_channel_toxicity', 
                          'like_count': 'avg_comment_likes'}).reset_index()
        
        user_map = pd.merge(users_df[['user_id', 'login_name', 'display_name']], map_df, 
                           left_on='login_name', right_on='twitch_login_name')
        
        combined_metrics = pd.merge(twitch_metrics, user_map, on='user_id')
        combined_metrics = pd.merge(combined_metrics, yt_metrics, left_on='youtube_channel_id', right_on='channel_id')
        
        if combined_metrics.empty:
            st.error("Insufficient overlapping data.")
        else:
            x_metric = st.sidebar.selectbox("X-Axis (Twitch)", ["avg_viewers", "stream_count"])
            y_metric = st.sidebar.selectbox("Y-Axis (YouTube)", ["total_comments", "avg_channel_toxicity", "avg_comment_likes"])
            
            valid_data = combined_metrics[[x_metric, y_metric]].dropna()
            
            if len(valid_data) < 3:
                st.error(f"Insufficient valid data. Only {len(valid_data)} creators have both metrics.")
            else:
                plot_data = combined_metrics.loc[valid_data.index]
                
                pearson_r, _ = stats.pearsonr(plot_data[x_metric], plot_data[y_metric])
                spearman_r, _ = stats.spearmanr(plot_data[x_metric], plot_data[y_metric])
                slope, intercept, r_value, p_value, _ = stats.linregress(plot_data[x_metric], plot_data[y_metric])
                
                display_stats(pearson_r, spearman_r, slope, r_value**2, p_value)
                
                st.subheader(f"{x_metric} vs. {y_metric}")
                
                scatter = alt.Chart(plot_data).mark_circle(size=60).encode(
                    x=alt.X(x_metric, scale=alt.Scale(zero=False)),
                    y=alt.Y(y_metric, scale=alt.Scale(zero=False)),
                    tooltip=['display_name', x_metric, y_metric]
                ).interactive()
                
                reg_line = scatter.transform_regression(x_metric, y_metric).mark_line(color='red')
                st.altair_chart(scatter + reg_line, width='stretch')
                
                st.markdown("### Creator Data")
                st.dataframe(plot_data[['display_name', x_metric, y_metric]].sort_values(x_metric, ascending=False))

# --- PAGE 4: RQ3 (CONTENT THEMES) ---
elif page == "RQ3: Content Themes":
    st.title("RQ3: Content Theme Analyzer")
    st.markdown("**Question:** How do specific keywords influence engagement?")
    
    keyword = st.sidebar.text_input("Keyword", "drama").lower()
    
    with st.spinner("Analyzing content themes..."):
        videos_df = load_videos_data()
        comment_counts = get_comment_counts()
    
    df = videos_df.copy()
    df['has_keyword'] = df['video_title'].str.lower().str.contains(keyword, na=False)
    df = pd.merge(df, comment_counts, on='video_id', how='left')
    df['comment_count'] = df['comment_count'].fillna(0)
    
    if df['has_keyword'].sum() > 0:
        group_yes = df[df['has_keyword']]['comment_count']
        group_no = df[~df['has_keyword']]['comment_count']
        
        st.markdown(f"### Statistical Test: Does '{keyword}' drive engagement?")
        
        t_stat, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
        
        m1, m2 = group_yes.mean(), group_no.mean()
        lift = ((m1 - m2) / m2) * 100 if m2 > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Comments (With)", f"{m1:.1f}")
        c2.metric("Avg Comments (Without)", f"{m2:.1f}")
        c3.metric("Engagement Lift", f"{lift:+.1f}%")
        
        if p_val < 0.05:
            st.success(f"**Statistically Significant** (p = {p_val:.4e})")
        else:
            st.warning(f"**Not Significant** (p = {p_val:.4f})")

        st.subheader("Engagement Distribution")
        sample_size = min(3000, len(df))
        chart_data = df[['has_keyword', 'comment_count']].sample(n=sample_size, random_state=42)
        chart_data['Type'] = chart_data['has_keyword'].map({True: 'With Keyword', False: 'Without Keyword'})
        
        chart = alt.Chart(chart_data).transform_density(
            'comment_count', as_=['comment_count', 'density'], groupby=['Type']
        ).mark_area(opacity=0.5).encode(x='comment_count:Q', y='density:Q', color='Type:N')
        st.altair_chart(chart, width='stretch')
        
        st.subheader(f"Co-occurring Keywords with '{keyword}'")
        try:
            subset = df[df['has_keyword']]['video_title'].dropna()
            if len(subset) > 0:
                vec = CountVectorizer(stop_words='english', max_features=15)
                bow = vec.fit_transform(subset.astype(str))
                words = pd.DataFrame({'word': vec.get_feature_names_out(), 'count': bow.toarray().sum(axis=0)})
                words = words[words['word'] != keyword].sort_values('count', ascending=False)
                
                bar = alt.Chart(words).mark_bar().encode(x='count:Q', y=alt.Y('word:N', sort='-x'))
                st.altair_chart(bar, width='stretch')
        except:
            st.info("Not enough data for keyword analysis.")

        st.subheader("Example Videos")
        st.dataframe(df[df['has_keyword']][['video_title', 'video_id']].head(10))
    else:
        st.error("Keyword not found in any titles.")