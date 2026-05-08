import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

st.set_page_config(page_title="EduPro Personalization", layout="wide")

st.title("🎓 EduPro Personalization Engine")

# SAMPLE DATA GENERATION (if files don't exist)

@st.cache_data
def generate_sample_data():
    """Generate sample data if CSV files don't exist"""
    np.random.seed(42)
    
    # Users data
    users = pd.DataFrame({
        'UserID': range(1, 101),
        'Age': np.random.randint(18, 65, 100),
        'total_courses': np.random.randint(1, 50, 100),
        'avg_spending': np.random.uniform(10, 500, 100),
        'Cluster': np.random.choice([0,1,2,3], 100)
    })
    
    # Courses data
    courses = pd.DataFrame({
        'CourseID': range(1, 51),
        'CourseCategory': np.random.choice(['Data Science', 'Web Dev', 'Marketing', 'Design', 'Business'], 50),
        'CourseLevel': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 50),
        'CourseRating': np.random.uniform(3.5, 5.0, 50)
    })
    
    # Transactions data
    transactions = pd.DataFrame({
        'UserID': np.random.choice(users['UserID'], 1000),
        'CourseID': np.random.choice(courses['CourseID'], 1000),
        'Amount': np.random.uniform(10, 200, 1000),
        'Date': pd.date_range('2023-01-01', periods=1000, freq='D')
    })
    
    # Final DF with user features
    final_df = users.copy()
    
    return users, courses, transactions, final_df


# LOAD DATA

@st.cache_data
def load_data():
    try:
        users = pd.read_csv("users.csv")
        courses = pd.read_csv("courses.csv")
        transactions = pd.read_csv("transactions.csv")
        final_df = pd.read_csv("final_df.csv")
        st.success("✅ Data loaded successfully!")
        return users, courses, transactions, final_df
    except FileNotFoundError:
        st.warning("⚠️ CSV files not found. Generating sample data...")
        return generate_sample_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None

users, courses, transactions, final_df = load_data()

if final_df is None:
    st.stop()


# SIDEBAR

st.sidebar.header("🔍 Select Learner")

# Ensure UserID column exists and get unique values
if 'UserID' not in final_df.columns:
    st.error("❌ UserID column missing in final_df.csv")
    st.stop()

user_ids = sorted(final_df['UserID'].unique())
user_id = st.sidebar.selectbox("Choose UserID", user_ids)

user_data = final_df[final_df['UserID'] == user_id].iloc[0] if not final_df[final_df['UserID'] == user_id].empty else None

if user_data is None:
    st.error("❌ User data not found")
    st.stop()

# USER PROFILE

st.subheader("👤 Learner Profile")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🆔 User ID", user_id)
with col2:
    st.metric("📚 Total Courses", int(user_data.get('total_courses', 0)))
with col3:
    st.metric("💰 Avg Spending", f"${user_data.get('avg_spending', 0):.2f}")


# CLUSTER INFO

if 'Cluster' not in final_df.columns:
    st.error("❌ Cluster column missing in final_df.csv")
    st.stop()

cluster = int(user_data['Cluster'])
cluster_names = {
    0: "Explorer 🌍 - Loves diverse topics",
    1: "Specialist 🎯 - Deep dives in one area", 
    2: "Casual Learner 😌 - Occasional learning",
    3: "High Spender 💰 - Invests in premium courses"
}

st.subheader("🧠 Learner Segment")
st.success(f"**{cluster_names.get(cluster, f'Cluster {cluster}')}**")


# CLUSTER DISTRIBUTION

st.subheader("📊 Cluster Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
cluster_counts = final_df['Cluster'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
cluster_counts.plot(kind='bar', ax=ax, color=colors)
ax.set_title('Learner Distribution by Cluster')
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Learners')
plt.xticks(rotation=0)
st.pyplot(fig)


# # RECOMMENDATION ENGINE

st.subheader("🎯 Personalized Course Recommendations")

# Prepare data for recommendations
df = transactions.merge(courses, on='CourseID')
df = df.merge(final_df[['UserID', 'Cluster']], on='UserID', how='left')

# Filter by cluster
cluster_courses = df[df['Cluster'] == cluster].copy()

# Remove already taken courses
user_courses = set(transactions[transactions['UserID'] == user_id]['CourseID'].unique())
cluster_courses = cluster_courses[~cluster_courses['CourseID'].isin(user_courses)]

if cluster_courses.empty:
    st.warning("No new recommendations available for this cluster.")
else:
    # Generate recommendations based on cluster popularity
    recommendations = (
        cluster_courses.groupby(['CourseID', 'CourseCategory', 'CourseLevel', 'CourseRating'])
        .size()
        .reset_index(name='popularity_score')
        .sort_values('popularity_score', ascending=False)
        .head(10)
    )
    
    st.dataframe(recommendations[['CourseID', 'CourseCategory', 'CourseLevel', 'CourseRating', 'popularity_score']], 
                use_container_width=True)


# FILTERS

st.sidebar.header("🎛️ Filters")

category_options = ["All"] + sorted(courses['CourseCategory'].dropna().unique().tolist())
level_options = ["All"] + sorted(courses['CourseLevel'].dropna().unique().tolist())

category_filter = st.sidebar.selectbox("📂 Category", category_options)
level_filter = st.sidebar.selectbox("📊 Level", level_options)

# Apply filters to recommendations
if 'recommendations' in locals():
    filtered_recs = recommendations.copy()
    
    if category_filter != "All":
        filtered_recs = filtered_recs[filtered_recs['CourseCategory'] == category_filter]
    
    if level_filter != "All":
        filtered_recs = filtered_recs[filtered_recs['CourseLevel'] == level_filter]
    
    st.subheader("🔎 Filtered Recommendations")
    if filtered_recs.empty:
        st.info("No courses match the selected filters.")
    else:
        st.dataframe(filtered_recs, use_container_width=True)


# SEGMENT ANALYSIS

st.subheader("📊 Segment Comparison")

numeric_cols = final_df.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 1:  # Ensure we have data beyond just Cluster
    analysis_cols = [col for col in numeric_cols if col != 'Cluster' and col != 'UserID']
    if analysis_cols:
        comparison = final_df[analysis_cols + ['Cluster']].groupby('Cluster').mean()
        st.dataframe(comparison.round(2), use_container_width=True)


# VISUALIZATIONS

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Spending Distribution")
    fig, ax = plt.subplots()
    final_df.boxplot(column='avg_spending', by='Cluster', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📚 Course Engagement")
    fig, ax = plt.subplots()
    final_df.boxplot(column='total_courses', by='Cluster', ax=ax)
    st.pyplot(fig)

# INSIGHTS

st.markdown("---")
st.markdown("### 🚀 Actionable Insights")

cluster_insights = {
    0: "🌍 **Explorer**: Recommend diverse course bundles across multiple categories",
    1: "🎯 **Specialist**: Suggest advanced courses and learning paths in their focus area", 
    2: "😌 **Casual Learner**: Offer short courses and micro-learning content",
    3: "💰 **High Spender**: Promote premium certifications and exclusive content"
}

st.info(cluster_insights.get(cluster, "Consider personalized outreach strategies."))

# Footer
st.markdown("---")
st.markdown("* EduPro Personalization Engine*")