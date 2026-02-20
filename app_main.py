import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="eVidyaloka Impact Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: white; padding: 15px; border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #00796b;
    }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .summary-box {
        background-color: #e0f2f1; border-left: 6px solid #00796b;
        padding: 20px; border-radius: 5px; margin-top: 20px; color: #004d40;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. FALLBACK INFERENCE ENGINE (Only used if 'Skillwise Mapping' is missing)
# -----------------------------------------------------------------------------

def infer_skill_from_text(question_text):
    if pd.isna(question_text): return "Uncategorized"
    text = str(question_text).lower()

    # --- MATH ---
    if any(x in text for x in ['add', 'sum', 'total', 'subtract', 'minus', 'plus']): return "Math: Arithmetic"
    if any(x in text for x in ['multiply', 'product', 'divide', 'share']): return "Math: Operations"
    if any(x in text for x in ['shape', 'circle', 'angle', 'area']): return "Math: Geometry"

    # --- SCIENCE ---
    if any(x in text for x in ['plant', 'leaf', 'root', 'flower']): return "Bio: Plants"
    if any(x in text for x in ['body', 'organ', 'digest', 'breath']): return "Bio: Human Body"

    # --- ENGLISH ---
    if any(x in text for x in ['spelling', 'spell']): return "Eng: Spelling"
    if any(x in text for x in ['noun', 'pronoun', 'adjective', 'verb']): return "Eng: Parts of Speech"
    if any(x in text for x in ['tense', 'past', 'present']): return "Eng: Tenses"
    if any(x in text for x in ['article', 'a, an, the']): return "Eng: Articles"

    return "General Application"


# -----------------------------------------------------------------------------
# 3. DATA PROCESSING
# -----------------------------------------------------------------------------

def extract_response_details(val):
    try:
        if pd.isna(val): return None, None
        clean_val = str(val).strip()
        if clean_val.startswith('"') and clean_val.endswith('"'):
            clean_val = clean_val[1:-1].replace('""', '"')
        data = json.loads(clean_val)
        return int(data.get('value', -1)), str(data.get('text', ''))
    except:
        return None, None


@st.cache_data
def load_and_parse_workbook(uploaded_file):
    try:
        xls = pd.read_excel(uploaded_file, sheet_name=None)
        sheet_names = list(xls.keys())

        subjects = {}
        for name in sheet_names:
            if 'Baseline' in name:
                parts = re.split(r'[-_\s]', name)
                try:
                    idx = next(i for i, part in enumerate(parts) if 'Baseline' in part)
                    subject = parts[idx + 1] if idx + 1 < len(parts) else "General"
                except:
                    subject = "General"

                endline_match = next((s for s in sheet_names if 'Endline' in s and subject in s), None)
                if endline_match:
                    subjects[subject] = {'Baseline': name, 'Endline': endline_match}

        if not subjects:
            bl = next((s for s in sheet_names if 'Baseline' in s), None)
            el = next((s for s in sheet_names if 'Endline' in s), None)
            if bl and el:
                subjects["Standard"] = {'Baseline': bl, 'Endline': el}

        key_sheet = next((s for s in sheet_names if 'Key' in s or 'Answer' in s), None)

        return xls, subjects, key_sheet

    except Exception as e:
        st.error(f"Error loading workbook: {e}")
        return None, {}, None


def process_subject_data(xls, subject_config, key_sheet_name):
    try:
        baseline_df = xls[subject_config['Baseline']]
        endline_df = xls[subject_config['Endline']]

        if not key_sheet_name:
            st.error("No Answer Key sheet found!")
            return None, None, None, None

        answer_key = xls[key_sheet_name].copy()

        # --- SKILL MAPPING LOGIC ---
        answer_key.columns = answer_key.columns.str.strip()

        if 'Skillwise Mapping' in answer_key.columns:
            answer_key['Skill'] = answer_key['Skillwise Mapping'].astype(str).str.strip()
            answer_key['Skill'] = answer_key['Skill'].replace(
                {'nan': 'Uncategorized', 'NaN': 'Uncategorized', '': 'Uncategorized'})
        elif 'Skill' in answer_key.columns:
            answer_key['Skill'] = answer_key['Skill']
        else:
            if 'Question Text' in answer_key.columns:
                answer_key['Skill'] = answer_key['Question Text'].apply(infer_skill_from_text)
            else:
                answer_key['Skill'] = "Uncategorized"

        def process_sheet(df, assessment_type):
            def calc_row(row):
                score, max_score = 0, 0
                grade = row['Grade']
                g_str = str(grade).replace('G', '').strip()

                key_subset = answer_key[answer_key['Grade'].astype(str).str.contains(g_str, na=False)]
                key_subset = key_subset[key_subset['Assessment'] == assessment_type]

                text_responses = {}

                for q_num in range(1, 11):
                    q_col = f"Q{q_num}"
                    if q_col in row:
                        val, text = extract_response_details(row[q_col])
                        text_responses[f"{q_col}_Text"] = text

                        q_key = key_subset[key_subset['Question #'] == q_num]
                        if not q_key.empty and val is not None:
                            correct_val = q_key.iloc[0]['Correct Value']
                            if val == correct_val: score += 1
                            max_score += 1

                return score, max_score, text_responses

            results = df.apply(calc_row, axis=1, result_type='expand')
            df['Score'] = results[0]
            df['Max_Score'] = results[1]
            df['Percentage'] = (df['Score'] / df['Max_Score']) * 100

            text_df = pd.DataFrame(results[2].tolist())
            df = pd.concat([df, text_df], axis=1)
            return df

        baseline_df = process_sheet(baseline_df, 'Baseline')
        endline_df = process_sheet(endline_df, 'Endline')

        cols_base = ['Student ID', 'State', 'Center', 'Grade', 'Percentage', 'Score']
        merged_df = pd.merge(
            baseline_df[cols_base],
            endline_df[['Student ID', 'Percentage', 'Score']],
            on='Student ID', suffixes=('_BL', '_EL')
        )
        merged_df['Growth'] = merged_df['Percentage_EL'] - merged_df['Percentage_BL']

        return merged_df, baseline_df, endline_df, answer_key

    except Exception as e:
        st.error(f"Error processing subject data: {e}")
        return None, None, None, None


# -----------------------------------------------------------------------------
# 4. AGGREGATION FUNCTIONS
# -----------------------------------------------------------------------------

def compute_skill_stats(df, key_df, assessment_type):
    stats = []
    for grade in df['Grade'].unique():
        grade_df = df[df['Grade'] == grade]
        g_str = str(grade).replace('G', '').strip()

        key_subset = key_df[key_df['Grade'].astype(str).str.contains(g_str, na=False)]
        key_subset = key_subset[key_subset['Assessment'] == assessment_type]

        for q_num in range(1, 11):
            q_col = f"Q{q_num}"
            if q_col in grade_df.columns:
                q_key = key_subset[key_subset['Question #'] == q_num]
                if not q_key.empty:
                    skill = q_key.iloc[0]['Skill']
                    correct_val = q_key.iloc[0]['Correct Value']

                    values = grade_df[q_col].apply(lambda x: extract_response_details(x)[0])
                    valid_values = values[values.notna() & (values != -1)]
                    if not valid_values.empty:
                        accuracy = (valid_values == correct_val).mean() * 100
                        stats.append({'Skill': skill, 'Accuracy': accuracy, 'Count': len(valid_values)})

    if not stats: return pd.DataFrame()
    return pd.DataFrame(stats).groupby('Skill')['Accuracy'].mean().reset_index()


def compute_errors(df, key_df, assessment_type):
    errors = []
    for grade in df['Grade'].unique():
        grade_df = df[df['Grade'] == grade]
        g_str = str(grade).replace('G', '').strip()
        key_subset = key_df[key_df['Grade'].astype(str).str.contains(g_str, na=False)]
        key_subset = key_subset[key_subset['Assessment'] == assessment_type]

        for q_num in range(1, 11):
            q_col = f"Q{q_num}"
            if q_col in grade_df.columns:
                q_key = key_subset[key_subset['Question #'] == q_num]
                if not q_key.empty:
                    correct_val = q_key.iloc[0]['Correct Value']
                    skill = q_key.iloc[0]['Skill']

                    values = grade_df[q_col].apply(lambda x: extract_response_details(x)[0])
                    texts = grade_df[q_col].apply(lambda x: extract_response_details(x)[1])

                    wrong_texts = texts[(values != correct_val) & (values != -1) & (values.notna())]

                    if not wrong_texts.empty:
                        top_error = wrong_texts.mode()[0]
                        freq = wrong_texts.value_counts().iloc[0]
                        errors.append({'Skill': skill, 'Common Error': top_error, 'Count': freq})
    return pd.DataFrame(errors)


# -----------------------------------------------------------------------------
# 5. MAIN UI
# -----------------------------------------------------------------------------
st.sidebar.title("📂 Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload Workbook (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls_data, subjects, key_sheet = load_and_parse_workbook(uploaded_file)

    if subjects:
        if len(subjects) > 1:
            st.sidebar.markdown("---")
            selected_subject = st.sidebar.selectbox("📚 Select Subject", list(subjects.keys()))
        else:
            selected_subject = list(subjects.keys())[0]
            st.sidebar.success(f"Loaded Subject: {selected_subject}")

        df_merged, df_bl, df_el, answer_key = process_subject_data(xls_data, subjects[selected_subject], key_sheet)

        if df_merged is not None:
            # Filters
            st.sidebar.markdown("---")
            st.sidebar.header("🔍 Filters")
            sel_state = st.sidebar.multiselect("State", df_merged['State'].unique(),
                                               default=df_merged['State'].unique())
            sel_grade = st.sidebar.multiselect("Grade", sorted(df_merged['Grade'].unique()),
                                               default=sorted(df_merged['Grade'].unique()))

            mask = (df_merged['State'].isin(sel_state)) & (df_merged['Grade'].isin(sel_grade))
            f_merged = df_merged[mask]
            f_bl = df_bl[(df_bl['Student ID'].isin(f_merged['Student ID']))]
            f_el = df_el[(df_el['Student ID'].isin(f_merged['Student ID']))]

            st.title(f"Impact Dashboard: {selected_subject} 🚀")
            tabs = st.tabs(["📈 Executive Summary", "🏫 Centre Performance", "🔍 Student Deep Dive", "❓ Question Analysis",
                            "🧠 Skill Analysis"])

            # Tab 1
            with tabs[0]:
                st.markdown("### High Level Overview")
                avg_bl, avg_el = f_merged['Percentage_BL'].mean(), f_merged['Percentage_EL'].mean()
                imp_count = f_merged[f_merged['Growth'] > 0].shape[0]

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Students", len(f_merged))
                k2.metric("Baseline", f"{avg_bl:.1f}%")
                k3.metric("Endline", f"{avg_el:.1f}%", delta=f"{avg_el - avg_bl:.1f}%")
                k4.metric("Growth", f"{f_merged['Growth'].mean():.1f}%")
                st.markdown(
                    f"""
                    <div class="summary-box">
                        <b>📊 Detailed Insights & Metric Explanations:</b><br><br>
                        <ul>
                            <li><b>Students:</b> The total number of students ({len(f_merged)}) who participated in both the Baseline and Endline assessments based on your current filters.</li>
                            <li><b>Baseline:</b> The average score ({avg_bl:.1f}%) achieved by these students at the beginning of the program.</li>
                            <li><b>Endline:</b> The average score ({avg_el:.1f}%) achieved at the end of the program. The delta shows the overall shift.</li>
                            <li><b>Growth:</b> The average individual student improvement ({f_merged['Growth'].mean():.1f}%).</li>
                        </ul>
                        <b>Key Takeaway:</b> Out of {len(f_merged)} students, <b>{imp_count} students ({imp_count / len(f_merged) * 100:.1f}%)</b> showed positive improvement in their scores from Baseline to Endline.
                    </div>
                    """,
                    unsafe_allow_html=True)

            # Tab 2
            with tabs[1]:
                st.markdown("### Centre Performance")
                # Aggregation to calculate Student Count
                c_stats = f_merged.groupby('Center').agg(
                    Growth=('Growth', 'mean'),
                    Percentage_EL=('Percentage_EL', 'mean'),
                    Student_Count=('Student ID', 'count')
                ).reset_index()

                fig = px.scatter(
                    c_stats,
                    x='Percentage_EL',
                    y='Growth',
                    hover_name='Center',
                    color='Growth',
                    title="Growth vs Proficiency (Size = Student Count)",
                    color_continuous_scale='Teal',
                    size='Student_Count',  # Relative size
                    size_max=60
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    """
                    <div class="summary-box" style="margin-bottom: 10px;">
                        <b>📖 How to Read This Graph:</b><br>
                        <ul>
                            <li><b>X-Axis (Proficiency):</b> Shows the average Endline score. Centres further to the right have higher final scores.</li>
                            <li><b>Y-Axis (Growth):</b> Shows the average improvement from Baseline. Centres higher up showed greater improvement.</li>
                            <li><b>Bubble Size:</b> Represents the number of students at the centre. Larger bubbles mean more students took the assessment.</li>
                            <li><b>Color:</b> Darker colors (more teal) indicate higher overall growth.</li>
                            <li><b>Ideal Position:</b> Top-Right corner (Centres that achieved high final scores AND high improvement).</li>
                        </ul>
                    </div>

                    <div class="summary-box">
                        <b>🔑 Key Metrics Explained:</b><br>
                        <ul>
                            <li><b>Percentage_EL:</b> The average percentage scored by all students in the Endline assessment for a specific centre.</li>
                            <li><b>Growth:</b> The average difference between Endline and Baseline scores for students in that centre.</li>
                            <li><b>Student Count:</b> The total number of students who completed both the Baseline and Endline assessments at that centre.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Tab 3
            with tabs[2]:
                st.markdown("### Student List")
                search = st.text_input("Search Student ID")
                show = f_merged[f_merged['Student ID'].astype(str).str.contains(search)] if search else f_merged
                st.dataframe(show[['Student ID', 'Grade', 'Center', 'Percentage_BL', 'Percentage_EL', 'Growth']],
                             use_container_width=True)

            # Tab 4
            with tabs[3]:
                st.markdown("### Question Difficulty Heatmap")

                # Dropdown menu to select assessment type
                assessment_choice = st.selectbox("Select Assessment to View:", ["Baseline", "Endline"], index=1)

                # Select the correct dataframe based on user choice
                target_df = f_bl if assessment_choice == "Baseline" else f_el

                q_data = []
                for g in target_df['Grade'].unique():
                    g_df = target_df[target_df['Grade'] == g]
                    g_str = str(g).replace('G', '').strip()
                    k_sub = answer_key[answer_key['Grade'].astype(str).str.contains(g_str, na=False)]
                    k_sub = k_sub[k_sub['Assessment'] == assessment_choice]

                    for q in range(1, 11):
                        q_k = k_sub[k_sub['Question #'] == q]
                        if not q_k.empty:
                            corr = q_k.iloc[0]['Correct Value']
                            vals = g_df[f"Q{q}"].apply(lambda x: extract_response_details(x)[0])
                            valid = vals[vals != -1]
                            if not valid.empty:
                                acc = (valid == corr).mean() * 100
                                q_data.append({'Grade': g, 'Question': f"Q{q}", 'Accuracy': acc})

                if q_data:
                    fig_hm = px.density_heatmap(pd.DataFrame(q_data), x='Question', y='Grade', z='Accuracy',
                                                range_color=[0, 100], color_continuous_scale='RdYlGn', text_auto='.0f',
                                                title=f"{assessment_choice} Question Accuracy Heatmap")
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info(f"No data available for {assessment_choice} questions.")

            # Tab 5 (UPDATED WITH MISCONCEPTIONS DROPDOWN)
            with tabs[4]:
                st.markdown(f"### 🧠 Skill Analysis ({selected_subject})")

                if 'Skillwise Mapping' in answer_key.columns:
                    st.success("✅ Using explicit 'Skillwise Mapping' from Answer Key.")
                else:
                    st.info("ℹ️ Using automated keyword detection.")

                s_bl = compute_skill_stats(f_bl, answer_key, 'Baseline')
                s_el = compute_skill_stats(f_el, answer_key, 'Endline')

                if not s_bl.empty and not s_el.empty:
                    comp = pd.merge(s_bl, s_el, on='Skill', suffixes=('_BL', '_EL'))
                    comp = comp.sort_values('Accuracy_EL')

                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                        fig_s = go.Figure()
                        fig_s.add_trace(
                            go.Bar(name='Baseline', x=comp['Skill'], y=comp['Accuracy_BL'], marker_color='#bdc3c7'))
                        fig_s.add_trace(
                            go.Bar(name='Endline', x=comp['Skill'], y=comp['Accuracy_EL'], marker_color='#00796b'))

                        fig_s.update_layout(
                            title="Skill Proficiency Growth",
                            barmode='group',
                            height=500,
                            xaxis_title="Skill",
                            yaxis_title="Accuracy %",
                            legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
                        )
                        st.plotly_chart(fig_s, use_container_width=True)

                    with c2:
                        misconception_choice = st.selectbox("Select Assessment for Misconceptions:",
                                                            ["Baseline", "Endline"], index=1)
                        st.write(f"#### ⚠️ Common Misconceptions in {misconception_choice}")

                        target_err_df = f_bl if misconception_choice == "Baseline" else f_el
                        errs = compute_errors(target_err_df, answer_key, misconception_choice)

                        if not errs.empty:
                            st.dataframe(errs[['Skill', 'Common Error', 'Count']], hide_index=True)
                        else:
                            st.info(f"No significant common errors detected in {misconception_choice}.")
                else:
                    st.warning("Not enough data to calculate skill metrics.")
    else:
        st.error("Could not auto-detect subjects. Please ensure sheet names contain 'Baseline' and 'Endline'.")

else:
    st.info("Please upload your workbook.")
