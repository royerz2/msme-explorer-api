from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client if API key is available
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Load the dataset
data_file = os.getenv('DATA_FILE', 'data.csv')
df = pd.read_csv(data_file)

# Enable detailed logging if configured
ENABLE_DETAILED_LOGGING = os.getenv('ENABLE_DETAILED_LOGGING', 'false').lower() == 'true'
if ENABLE_DETAILED_LOGGING:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Detailed logging enabled")

# Define survey columns and other column groups
SURVEY_COLS = ['AU', 'INN', 'RT', 'PA', 'CA', 'OEO', 'OPC', 'RC', 'CCC', 'ORC', 'STC', 'CMC', 'OEC', 'SU', 'SY', 'CO', 'REO', 'OSRS', 'IA', 'II']
IT_COLS = ['IT_SM', 'IT_CS', 'IT_PD', 'IT_DM', 'IT_KM', 'IT_SCM', 'ODTA']
DEMOGRAPHIC_COLS = ['Gender', 'Respondent Age', 'Education', 'Position', 'Working period', 'City', 'Business fields', 'Current active workforce', 'MSME age']
PARTNERSHIP_COLS = ['Double partnership (DP)', 'Triple partnership (TP)']

# Abbreviation mappings
ABBREVIATION_MAPPINGS = {
    "AU": "Autonomy",
    "INN": "Innovation",
    "RT": "Risk_Taking",
    "PA": "Proactiveness",
    "CA": "Competitive_Aggressiveness",
    "OEO": "Entrepreneurial_Orientation",
    "OPC": "Opportunities_Creation",
    "RC": "Relationship_Capabilities",
    "CCC": "Creative_Capabilities",
    "ORC": "Organizational_Resources_Coordination",
    "STC": "Strategic_Coordination",
    "CMC": "Commitment_Capability",
    "OEC": "Entrepreneurial_Competence",
    "SU": "Crisis_Solution_Capability",
    "SY": "Crisis_Resource_Access",
    "CO": "Digital_Connectivity",
    "REO": "SDG_Orientation",
    "OSRS": "MSME_Resilience",
    "IA": "Internet_Availability",
    "II": "IT_Device_Availability",
    "IT_SM": "Digital_Sales_Marketing",
    "IT_CS": "Digital_Customer_Service",
    "IT_PD": "Digital_Product_Development",
    "IT_DM": "Digital_Decision_Making",
    "IT_KM": "Digital_Knowledge_Management",
    "IT_SCM": "Digital_Supply_Chain_Management",
    "ODTA": "Digital_Adoption",

    # Partnership columns
    'DP': 'Double Partnership',
    'TP': 'Triple Partnership'
}

def safe_float(value):
    """Convert value to float, return None if NaN or invalid"""
    if pd.isna(value) or math.isnan(float(value)) or math.isinf(float(value)):
        return None
    return float(value)

def safe_stats(series):
    """Calculate safe statistics for a pandas series"""
    if len(series) == 0:
        return {
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
    
    return {
        'mean': safe_float(series.mean()),
        'median': safe_float(series.median()),
        'std': safe_float(series.std()),
        'min': safe_float(series.min()),
        'max': safe_float(series.max()),
        'count': len(series)
    }

def get_gpt_analysis(prompt, analysis_type="general"):
    """Get analysis from GPT"""
    if not client:
        return f"GPT analysis not available (no API key provided for {analysis_type})"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert business analyst specializing in MSME (Micro, Small, and Medium Enterprises) research. Provide insightful, actionable analysis based on the data presented."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting GPT analysis for {analysis_type}: {str(e)}"

def convert_abbreviations(text):
    """Convert abbreviations to their full names"""
    if text in ABBREVIATION_MAPPINGS:
        return ABBREVIATION_MAPPINGS[text]
    return text

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "total_records": len(df)})

@app.route('/api/composite-scores', methods=['GET'])
def get_composite_scores():
    """Get composite scores analysis with GPT insights"""
    # Define column groups
    entrepreneurial_cols = ['AU', 'INN', 'RT', 'PA', 'CA']
    capabilities_cols = ['OPC', 'RC', 'CCC', 'ORC', 'STC', 'CMC', 'OEC']
    collaboration_cols = ['SU', 'SY', 'CO', 'REO', 'OSRS', 'IA', 'II']
    it_cols = ['IT_SM', 'IT_CS', 'IT_PD', 'IT_DM', 'IT_KM', 'IT_SCM']
    
    # Calculate composite scores
    composite_scores = pd.DataFrame()
    composite_scores['EO_Score'] = df[entrepreneurial_cols].mean(axis=1)
    composite_scores['Capabilities_Score'] = df[capabilities_cols].mean(axis=1)
    composite_scores['Collaboration_Score'] = df[collaboration_cols].mean(axis=1)
    composite_scores['IT_Score'] = df[it_cols].mean(axis=1)
    
    # Calculate statistics
    stats_summary = composite_scores.describe()
    corr_matrix = composite_scores.corr()
    
    # Prepare response data
    response_data = {
        'scores': {
            'EO_Score': safe_stats(composite_scores['EO_Score']),
            'Capabilities_Score': safe_stats(composite_scores['Capabilities_Score']),
            'Collaboration_Score': safe_stats(composite_scores['Collaboration_Score']),
            'IT_Score': safe_stats(composite_scores['IT_Score'])
        },
        'correlations': {
            'variables': list(corr_matrix.columns),
            'matrix': corr_matrix.values.tolist()
        }
    }
    
    # Add GPT analysis if available
    if client:
        prompt = f"""
        Analyze these MSME composite scores:
        
        Statistical Summary:
        {stats_summary.to_string()}
        
        Correlation Matrix:
        {corr_matrix.to_string()}
        
        The scores represent:
        - EO_Score: Entrepreneurial Orientation (autonomy, innovation, risk-taking, proactiveness, competitive aggressiveness)
        - Capabilities_Score: Organizational capabilities (performance, resources, customer relations, operations, strategy, change management)
        - Collaboration_Score: Collaboration effectiveness (sustainability, synergy, collaboration, resource efficiency, innovation adoption)
        - IT_Score: Information Technology adoption across various business functions
        
        What insights can you derive about MSME performance patterns and relationships between these dimensions?
        """
        
        response_data['gpt_analysis'] = get_gpt_analysis(prompt, "composite_scores")
    
    return jsonify(response_data)

@app.route('/api/demographics', methods=['GET'])
def get_demographics():
    """Get demographic distribution analysis"""
    demographics = {}
    
    # Map API column names to frontend expected keys
    column_mapping = {
        'Gender': 'gender',
        'Respondent Age': 'respondent_age',
        'Education': 'education',
        'Position': 'position',
        'Business fields': 'business_field',
        'MSME age': 'business_age'
    }
    
    for col in DEMOGRAPHIC_COLS:
        if col in df.columns:
            value_counts = df[col].value_counts()
            frontend_key = column_mapping.get(col, col.lower().replace(' ', '_'))
            demographics[frontend_key] = {
                'labels': [convert_abbreviations(label) for label in value_counts.index.tolist()],
                'values': value_counts.values.tolist(),
                'percentages': (value_counts / len(df) * 100).round(2).tolist()
            }
    
    # Cross-tabulations
    cross_tabs = {}
    
    # Gender vs Position
    if 'Gender' in df.columns and 'Position' in df.columns:
        ct = pd.crosstab(df['Gender'], df['Position'])
        cross_tabs['gender_vs_position'] = {
            'index': [convert_abbreviations(idx) for idx in ct.index.tolist()],
            'columns': [convert_abbreviations(col) for col in ct.columns.tolist()],
            'values': ct.values.tolist()
        }
    
    # Education vs Business fields
    if 'Education' in df.columns and 'Business fields' in df.columns:
        ct = pd.crosstab(df['Education'], df['Business fields'])
        cross_tabs['education_vs_business'] = {
            'index': [convert_abbreviations(idx) for idx in ct.index.tolist()],
            'columns': [convert_abbreviations(col) for col in ct.columns.tolist()],
            'values': ct.values.tolist()
        }
    
    return jsonify({
        'distributions': demographics,
        'cross_tabulations': cross_tabs
    })

@app.route('/api/survey-analysis', methods=['GET'])
def get_survey_analysis():
    """Get survey response analysis"""
    survey_data = df[SURVEY_COLS].copy()
    
    # Basic statistics
    basic_stats = {}
    for col in SURVEY_COLS:
        if col in survey_data.columns:
            basic_stats[convert_abbreviations(col)] = {
                'mean': float(survey_data[col].mean()),
                'median': float(survey_data[col].median()),
                'std': float(survey_data[col].std()),
                'min': float(survey_data[col].min()),
                'max': float(survey_data[col].max()),
                'distribution': survey_data[col].value_counts().sort_index().to_dict()
            }
    
    # Correlation matrix
    correlation_matrix = survey_data.corr().round(3)
    correlation_data = {
        'variables': [convert_abbreviations(var) for var in correlation_matrix.index.tolist()],
        'matrix': correlation_matrix.values.tolist()
    }
    
    # Top correlations
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_val = correlation_matrix.iloc[i, j]
            corr_pairs.append({
                'var1': convert_abbreviations(var1),
                'var2': convert_abbreviations(var2),
                'correlation': float(corr_val)
            })
    
    # Sort by absolute correlation value
    top_correlations = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]
    
    return jsonify({
        'basic_statistics': basic_stats,
        'correlation_matrix': correlation_data,
        'top_correlations': top_correlations
    })

@app.route('/api/comparative-analysis', methods=['GET'])
def get_comparative_analysis():
    """Get comparative analysis by demographics"""
    if df.empty:
        return jsonify({'error': 'No data available'}), 400
    
    available_survey_cols = [col for col in SURVEY_COLS if col in df.columns]
    if not available_survey_cols:
        return jsonify({'error': 'No survey columns found'}), 400
    
    comparisons = {}
    
    # Gender comparison
    if 'Gender' in df.columns:
        gender_comparison = {}
        gender_values = df['Gender'].unique()
        
        if len(gender_values) >= 2:
            for col in available_survey_cols:
                try:
                    male_scores = df[df['Gender'] == 'Male'][col].dropna()
                    female_scores = df[df['Gender'] == 'Female'][col].dropna()
                    
                    if len(male_scores) > 1 and len(female_scores) > 1:
                        # T-test with error handling
                        try:
                            t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
                            
                            gender_comparison[col] = {
                                'male_mean': safe_float(male_scores.mean()),
                                'female_mean': safe_float(female_scores.mean()),
                                'male_std': safe_float(male_scores.std()),
                                'female_std': safe_float(female_scores.std()),
                                'male_count': len(male_scores),
                                'female_count': len(female_scores),
                                't_statistic': safe_float(t_stat),
                                'p_value': safe_float(p_value),
                                'significant': bool(safe_float(p_value) is not None and safe_float(p_value) < 0.05) if safe_float(p_value) is not None else False
                            }
                        except Exception as e:
                            print(f"Error in t-test for {col}: {e}")
                            gender_comparison[col] = {
                                'male_mean': safe_float(male_scores.mean()),
                                'female_mean': safe_float(female_scores.mean()),
                                'male_std': safe_float(male_scores.std()),
                                'female_std': safe_float(female_scores.std()),
                                'male_count': len(male_scores),
                                'female_count': len(female_scores),
                                't_statistic': None,
                                'p_value': None,
                                'significant': False
                            }
                except Exception as e:
                    print(f"Error processing gender comparison for {col}: {e}")
        
        comparisons['gender'] = gender_comparison
    
    # Age group comparison
    if 'Respondent Age' in df.columns:
        age_comparison = {}
        age_groups = df['Respondent Age'].dropna().unique()
        
        for col in available_survey_cols:
            try:
                group_stats = []
                for age_group in age_groups:
                    group_data = df[df['Respondent Age'] == age_group][col].dropna()
                    if len(group_data) > 0:
                        stats_dict = safe_stats(group_data)
                        stats_dict['age_group'] = str(age_group)
                        group_stats.append(stats_dict)
                
                age_comparison[col] = group_stats
            except Exception as e:
                print(f"Error processing age comparison for {col}: {e}")
        
        comparisons['age_groups'] = age_comparison
    
    # Business field comparison
    if 'Business fields' in df.columns:
        business_comparison = {}
        business_fields = df['Business fields'].dropna().unique()
        
        for col in available_survey_cols:
            try:
                field_stats = []
                for field in business_fields:
                    field_data = df[df['Business fields'] == field][col].dropna()
                    if len(field_data) > 0:
                        stats_dict = safe_stats(field_data)
                        stats_dict['business_field'] = str(field)
                        field_stats.append(stats_dict)
                
                business_comparison[col] = field_stats
            except Exception as e:
                print(f"Error processing business comparison for {col}: {e}")
        
        comparisons['business_fields'] = business_comparison
    
    return jsonify(comparisons)

@app.route('/api/clustering', methods=['GET'])
def get_clustering():
    """Get clustering analysis (now only 2 clusters)"""
    try:
        # Prepare data for clustering
        cluster_data = df[SURVEY_COLS].dropna()
        
        if len(cluster_data) == 0:
            return jsonify({'error': 'No valid data for clustering'}), 400
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # K-means clustering with k=2 only
        clustering_results = {}
        k = 2
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Get cluster centers in original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Cluster statistics
        cluster_stats = []
        for i in range(k):
            cluster_mask = cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            
            cluster_profile = {}
            for j, col in enumerate(SURVEY_COLS):
                cluster_profile[col] = float(cluster_centers[i, j])
            
            cluster_stats.append({
                'cluster_id': i,
                'size': int(cluster_size),
                'profile': cluster_profile
            })
        
        clustering_results[f'k_{k}'] = {
            'clusters': cluster_stats,
            'inertia': float(kmeans.inertia_)
        }
        
        # Add demographics data for filters
        demographics = {
            'respondent_age': {
                'labels': df['Respondent Age'].unique().tolist(),
                'values': df['Respondent Age'].value_counts().tolist(),
                'percentages': (df['Respondent Age'].value_counts(normalize=True) * 100).round(2).tolist()
            },
            'working_period': {
                'labels': df['Working period'].unique().tolist(),
                'values': df['Working period'].value_counts().tolist(),
                'percentages': (df['Working period'].value_counts(normalize=True) * 100).round(2).tolist()
            }
        }
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        visualization_data = {
            'pca_components': pca_data.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
        
        return jsonify({
            'clustering_results': clustering_results,
            'visualization': visualization_data,
            'demographics': demographics
        })
        
    except Exception as e:
        print(f"Error in clustering analysis: {str(e)}")
        return jsonify({'error': f'Error in clustering analysis: {str(e)}'}), 500

@app.route('/api/technology-analysis', methods=['GET'])
def get_technology_analysis():
    """Get technology adoption analysis"""
    tech_data = df[IT_COLS].dropna()
    
    if len(tech_data) == 0:
        return jsonify({'error': 'No valid technology data'})
    
    # Basic tech statistics
    tech_stats = {}
    for col in IT_COLS:
        if col in df.columns:
            tech_stats[convert_abbreviations(col)] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'distribution': df[col].value_counts().sort_index().to_dict()
            }
    
    # Technology adoption by demographics
    tech_by_demographics = {}
    
    # By Gender
    if 'Gender' in df.columns:
        gender_tech = {}
        for col in IT_COLS:
            if col in df.columns:
                male_mean = df[df['Gender'] == 'Male'][col].mean()
                female_mean = df[df['Gender'] == 'Female'][col].mean()
                gender_tech[convert_abbreviations(col)] = {
                    'male': float(male_mean) if not pd.isna(male_mean) else 0,
                    'female': float(female_mean) if not pd.isna(female_mean) else 0
                }
        tech_by_demographics['gender'] = gender_tech
    
    # By Business Field
    if 'Business fields' in df.columns:
        business_tech = {}
        business_fields = df['Business fields'].unique()
        
        for field in business_fields:
            field_data = df[df['Business fields'] == field]
            field_tech = {}
            for col in IT_COLS:
                if col in df.columns:
                    mean_val = field_data[col].mean()
                    field_tech[convert_abbreviations(col)] = float(mean_val) if not pd.isna(mean_val) else 0
            business_tech[field] = field_tech
        
        tech_by_demographics['business_fields'] = business_tech
    
    # Technology correlation with business performance (using all survey scores as proxy)
    tech_performance_corr = {}
    performance_indicators = [col for col in SURVEY_COLS if col in df.columns]
    
    for tech_col in IT_COLS:
        if tech_col in df.columns:
            correlations = {}
            for perf_col in performance_indicators:
                if perf_col in df.columns:
                    corr = df[tech_col].corr(df[perf_col])
                    correlations[convert_abbreviations(perf_col)] = float(corr) if not pd.isna(corr) else 0
            tech_performance_corr[convert_abbreviations(tech_col)] = correlations
    
    return jsonify({
        'technology_statistics': tech_stats,
        'technology_by_demographics': tech_by_demographics,
        'technology_performance_correlation': tech_performance_corr
    })

@app.route('/api/partnership-analysis', methods=['GET'])
def get_partnership_analysis():
    """Get partnership analysis"""
    partnership_stats = {}
    
    # Partnership distribution
    for col in PARTNERSHIP_COLS:
        if col in df.columns:
            value_counts = df[col].value_counts()
            partnership_stats[col.lower().replace(' ', '_').replace('(', '').replace(')', '')] = {
                'labels': [convert_abbreviations(label) for label in value_counts.index.tolist()],
                'values': value_counts.values.tolist(),
                'percentages': (value_counts / len(df) * 100).round(2).tolist()
            }
    
    # Partnership impact on survey responses
    partnership_impact = {}
    
    for partnership_col in PARTNERSHIP_COLS:
        if partnership_col in df.columns:
            partnership_groups = df[partnership_col].unique()
            impact_data = {}
            
            for survey_col in SURVEY_COLS:
                if survey_col in df.columns:
                    group_means = []
                    for group in partnership_groups:
                        group_data = df[df[partnership_col] == group][survey_col].dropna()
                        if len(group_data) > 0:
                            group_means.append({
                                'partnership_type': convert_abbreviations(group),
                                'mean_score': float(group_data.mean()),
                                'count': len(group_data)
                            })
                    impact_data[convert_abbreviations(survey_col)] = group_means
            
            partnership_impact[partnership_col.lower().replace(' ', '_').replace('(', '').replace(')', '')] = impact_data
    
    return jsonify({
        'partnership_distribution': partnership_stats,
        'partnership_impact': partnership_impact
    })

@app.route('/api/comprehensive-report', methods=['GET'])
def get_comprehensive_report():
    """Get a comprehensive analysis report"""
    report = {}
    
    # Sample size and basic info
    report['sample_info'] = {
        'total_respondents': len(df),
        'complete_responses': len(df.dropna()),
        'survey_variables': len(SURVEY_COLS),
        'technology_variables': len(IT_COLS)
    }
    
    # Key findings
    key_findings = []
    
    # Gender distribution
    if 'Gender' in df.columns:
        gender_dist = df['Gender'].value_counts()
        key_findings.append({
            'category': 'Demographics',
            'finding': f"Gender distribution: {gender_dist.iloc[0]} {gender_dist.index[0]} ({gender_dist.iloc[0]/len(df)*100:.1f}%), {gender_dist.iloc[1]} {gender_dist.index[1]} ({gender_dist.iloc[1]/len(df)*100:.1f}%)"
        })
    
    # Most common business field
    if 'Business fields' in df.columns:
        top_business = df['Business fields'].value_counts().iloc[0:2]
        key_findings.append({
            'category': 'Business',
            'finding': f"Top business fields: {top_business.index[0]} ({top_business.iloc[0]} businesses), {top_business.index[1]} ({top_business.iloc[1]} businesses)"
        })
    
    # Highest scoring survey variables
    survey_means = df[SURVEY_COLS].mean().sort_values(ascending=False)
    key_findings.append({
        'category': 'Survey',
        'finding': f"Highest scoring variables: {survey_means.index[0]} ({survey_means.iloc[0]:.2f}), {survey_means.index[1]} ({survey_means.iloc[1]:.2f})"
    })
    
    # Technology adoption
    tech_means = df[IT_COLS].mean().sort_values(ascending=False)
    key_findings.append({
        'category': 'Technology',
        'finding': f"Highest technology adoption: {tech_means.index[0]} ({tech_means.iloc[0]:.2f}), {tech_means.index[1]} ({tech_means.iloc[1]:.2f})"
    })
    
    report['key_findings'] = key_findings
    
    return jsonify(report)

@app.route('/api/generate-pdf-report', methods=['GET'])
def generate_pdf_report():
    """Generate a comprehensive PDF report with visualizations and GPT analysis"""
    try:
        # Create a BytesIO buffer to store the PDF
        buffer = io.BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.8, 'MSME Analysis Report', ha='center', va='center', 
                    fontsize=24, fontweight='bold')
            fig.text(0.5, 0.7, 'AI-Powered Business Intelligence Analysis', ha='center', va='center', 
                    fontsize=16)
            fig.text(0.5, 0.6, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.5, f'Sample Size: {len(df)} MSMEs', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
            
            # Add key metrics
            fig.text(0.5, 0.4, 'Key Metrics Summary:', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
            
            # Calculate composite scores
            entrepreneurial_cols = ['AU', 'INN', 'RT', 'PA', 'CA']
            it_cols = ['IT_SM', 'IT_CS', 'IT_PD', 'IT_DM', 'IT_KM', 'IT_SCM']
            eo_score = df[entrepreneurial_cols].mean(axis=1).mean()
            it_score = df[it_cols].mean(axis=1).mean()
            odta_score = df['ODTA'].mean()
            
            fig.text(0.5, 0.35, f'Avg. Entrepreneurial Orientation: {eo_score:.2f}/7', 
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.32, f'Avg. IT Adoption: {it_score:.2f}/7', 
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.29, f'Avg. Digital Adoption: {odta_score:.2f}/7', 
                    ha='center', va='center', fontsize=12)
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add demographics analysis
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('MSME Demographics Analysis', fontsize=16, fontweight='bold')
            
            # Gender distribution
            gender_counts = df['Gender'].value_counts()
            axes[0,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[0,0].set_title('Gender Distribution')
            
            # Age distribution
            age_counts = df['Respondent Age'].value_counts()
            axes[0,1].bar(range(len(age_counts)), age_counts.values)
            axes[0,1].set_xticks(range(len(age_counts)))
            axes[0,1].set_xticklabels(age_counts.index, rotation=45)
            axes[0,1].set_title('Age Distribution')
            
            # Education distribution
            edu_counts = df['Education'].value_counts()
            axes[0,2].barh(range(len(edu_counts)), edu_counts.values)
            axes[0,2].set_yticks(range(len(edu_counts)))
            axes[0,2].set_yticklabels(edu_counts.index)
            axes[0,2].set_title('Education Distribution')
            
            # Business fields
            business_counts = df['Business fields'].value_counts()
            axes[1,0].pie(business_counts.values, labels=business_counts.index, autopct='%1.1f%%')
            axes[1,0].set_title('Business Fields Distribution')
            
            # City distribution
            city_counts = df['City'].value_counts()
            axes[1,1].bar(range(len(city_counts)), city_counts.values)
            axes[1,1].set_xticks(range(len(city_counts)))
            axes[1,1].set_xticklabels(city_counts.index, rotation=45)
            axes[1,1].set_title('City Distribution')
            
            # Workforce size
            workforce_counts = df['Current active workforce'].value_counts()
            axes[1,2].pie(workforce_counts.values, labels=workforce_counts.index, autopct='%1.1f%%')
            axes[1,2].set_title('Workforce Size Distribution')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add correlation heatmap
            fig, ax = plt.subplots(figsize=(16, 14))
            corr_matrix = df[SURVEY_COLS + IT_COLS].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('MSME Variables Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Add GPT analysis if available
            if client:
                # Executive summary
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(0.5, 0.95, 'Executive Summary', ha='center', va='top', 
                        fontsize=18, fontweight='bold')
                
                prompt = f"""
                Create an executive summary for this MSME analysis report:
                
                Sample Overview:
                - Total MSMEs analyzed: {len(df)}
                - Average Entrepreneurial Orientation Score: {eo_score:.2f}/7
                - Average IT Adoption Score: {it_score:.2f}/7
                - Average Digital Technology Adoption: {odta_score:.2f}/7
                
                Demographics:
                - Gender Distribution: {gender_counts.to_dict()}
                - Primary Business Fields: {business_counts.head(3).to_dict()}
                
                Key Correlations:
                - EO-ODTA Correlation: {df[entrepreneurial_cols].mean(axis=1).corr(df['ODTA']):.3f}
                - IT-ODTA Correlation: {df[it_cols].mean(axis=1).corr(df['ODTA']):.3f}
                
                Based on all the analyses performed, provide:
                1. Key findings and insights (3-4 main points)
                2. Strategic recommendations for MSME development
                3. Priority areas for intervention
                4. Implications for policy makers and business support organizations
                
                Keep it concise but comprehensive, suitable for senior executives and policy makers.
                """
                
                analysis = get_gpt_analysis(prompt, "executive_summary")
                fig.text(0.1, 0.85, analysis, ha='left', va='top', fontsize=10, 
                        wrap=True, transform=fig.transFigure)
                
                plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # Get the PDF content
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Convert to base64 for API response
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'message': 'PDF report generated successfully',
            'pdf_base64': pdf_base64
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating PDF report: {str(e)}'
        }), 500

@app.route('/api/correlational-analysis', methods=['GET'])
def get_correlational_analysis():
    """Get additional correlational analyses between different variable groups"""
    if df.empty:
        return jsonify({'error': 'No data available'}), 400

    results = {}

    # 1. IT Adoption vs Survey Variables
    available_it_cols = [col for col in IT_COLS if col in df.columns]
    available_survey_cols = [col for col in SURVEY_COLS if col in df.columns]

    if available_it_cols and available_survey_cols:
        it_survey_corr = df[available_it_cols + available_survey_cols].corr().loc[available_it_cols, available_survey_cols].round(3)
        results['it_survey_correlation'] = {
            'it_variables': [convert_abbreviations(var) for var in it_survey_corr.index.tolist()],
            'survey_variables': [convert_abbreviations(var) for var in it_survey_corr.columns.tolist()],
            'matrix': it_survey_corr.values.tolist()
        }

    # 2. Respondent Age vs IT Adoption
    if 'Respondent Age' in df.columns and available_it_cols:
        print("Processing Respondent Age vs IT Adoption correlation...")
        # Convert age groups to numeric values for correlation
        age_mapping = {age: i for i, age in enumerate(df['Respondent Age'].unique())}
        df['Respondent Age Numeric'] = df['Respondent Age'].map(age_mapping)
        
        age_it_corr = df[['Respondent Age Numeric'] + available_it_cols].corr().loc[['Respondent Age Numeric'], available_it_cols].round(3)
        results['age_it_correlation'] = {
            'age_variable': 'Respondent Age',
            'it_variables': [convert_abbreviations(var) for var in age_it_corr.columns.tolist()],
            'matrix': age_it_corr.values.tolist()
        }
        print("Respondent Age vs IT Adoption correlation completed")

    # 3. Company Age vs IT Adoption
    if 'MSME age' in df.columns and available_it_cols:
        print("Processing Company Age vs IT Adoption correlation...")
        # Convert company age to numeric values for correlation
        company_age_mapping = {age: i for i, age in enumerate(df['MSME age'].unique())}
        df['Company Age Numeric'] = df['MSME age'].map(company_age_mapping)
        
        company_age_it_corr = df[['Company Age Numeric'] + available_it_cols].corr().loc[['Company Age Numeric'], available_it_cols].round(3)
        results['company_age_it_correlation'] = {
            'age_variable': 'Company Age',
            'it_variables': [convert_abbreviations(var) for var in company_age_it_corr.columns.tolist()],
            'matrix': company_age_it_corr.values.tolist()
        }
        print("Company Age vs IT Adoption correlation completed")

    # 4. Company Age vs Business Performance
    if 'MSME age' in df.columns and available_survey_cols:
        print("Processing Company Age vs Business Performance correlation...")
        company_age_perf_corr = df[['Company Age Numeric'] + available_survey_cols].corr().loc[['Company Age Numeric'], available_survey_cols].round(3)
        results['company_age_performance_correlation'] = {
            'age_variable': 'Company Age',
            'performance_variables': [convert_abbreviations(var) for var in company_age_perf_corr.columns.tolist()],
            'matrix': company_age_perf_corr.values.tolist()
        }
        print("Company Age vs Business Performance correlation completed")

    # 5. Respondent Age vs Business Performance
    if 'Respondent Age' in df.columns and available_survey_cols:
        print("Processing Respondent Age vs Business Performance correlation...")
        age_perf_corr = df[['Respondent Age Numeric'] + available_survey_cols].corr().loc[['Respondent Age Numeric'], available_survey_cols].round(3)
        results['age_performance_correlation'] = {
            'age_variable': 'Respondent Age',
            'performance_variables': [convert_abbreviations(var) for var in age_perf_corr.columns.tolist()],
            'matrix': age_perf_corr.values.tolist()
        }
        print("Respondent Age vs Business Performance correlation completed")

    print("Final results:", results.keys())
    return jsonify(results)

@app.route('/api/filtered-pca', methods=['GET'])
def get_filtered_pca():
    """Get filtered PCA data for visualization"""
    filter_type = request.args.get('filter_type')
    filter_value = request.args.get('filter_value')
    
    if not filter_type or not filter_value:
        return jsonify({'error': 'Both filter_type and filter_value are required'}), 400
    
    # Prepare data for PCA
    cluster_data = df[SURVEY_COLS].copy()
    
    # Apply filter
    if filter_type == 'respondent_age':
        filtered_data = df[df['Respondent Age'] == filter_value]
    elif filter_type == 'working_period':
        # Handle special characters in working period values
        working_periods = df['Working period'].unique()
        # Find the exact match in the unique values
        matching_period = next((period for period in working_periods if str(period) == filter_value), None)
        if matching_period is None:
            return jsonify({'error': f'Invalid working period value: {filter_value}'}), 400
        filtered_data = df[df['Working period'] == matching_period]
    else:
        return jsonify({'error': 'Invalid filter_type. Must be either "respondent_age" or "working_period"'}), 400
    
    if len(filtered_data) == 0:
        return jsonify({'error': 'No data available for the selected filter'}), 400
    
    # Get filtered survey data
    filtered_survey_data = filtered_data[SURVEY_COLS].dropna()
    
    if len(filtered_survey_data) == 0:
        return jsonify({'error': 'No valid survey data after filtering'}), 400
    
    # Check if we have enough samples for clustering
    if len(filtered_survey_data) < 3:
        return jsonify({
            'error': f'Insufficient data for clustering. Found {len(filtered_survey_data)} samples, need at least 3.',
            'filter_info': {
                'type': filter_type,
                'value': filter_value,
                'sample_size': len(filtered_survey_data)
            }
        }), 400
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_survey_data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    # Get feature loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=SURVEY_COLS
    )
    
    # Get top features for each component
    top_features_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(5).to_dict()
    top_features_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(5).to_dict()
    
    # Perform clustering on filtered data
    kmeans = KMeans(n_clusters=min(3, len(filtered_survey_data)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Calculate cluster statistics
    cluster_stats = []
    for i in range(len(np.unique(cluster_labels))):
        cluster_mask = cluster_labels == i
        cluster_size = np.sum(cluster_mask)
        
        cluster_profile = {}
        for j, col in enumerate(SURVEY_COLS):
            cluster_profile[convert_abbreviations(col)] = float(filtered_survey_data.iloc[cluster_mask][col].mean())
        
        cluster_stats.append({
            'cluster_id': i,
            'size': int(cluster_size),
            'profile': cluster_profile
        })
    
    return jsonify({
        'pca_data': {
            'components': pca_data.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'loadings': {
                'PC1': top_features_pc1,
                'PC2': top_features_pc2
            }
        },
        'clustering': {
            'labels': cluster_labels.tolist(),
            'clusters': cluster_stats
        },
        'filter_info': {
            'type': filter_type,
            'value': filter_value,
            'sample_size': len(filtered_data)
        }
    })

@app.route('/api/download-raw-data', methods=['GET'])
def download_raw_data():
    """Serve the raw data.csv file for download."""
    app.logger.info("Attempting to serve data.csv")
    try:
        # Assuming data.csv is in the same directory as main.py or a known path
        file_path = os.path.join(app.root_path, 'data.csv')
        app.logger.info(f"Looking for file at: {file_path}")
        return send_file(file_path, mimetype='text/csv', as_attachment=True, download_name='data.csv')
    except FileNotFoundError:
        app.logger.error(f"data.csv not found at {file_path}")
        return jsonify({'error': 'data.csv not found.'}), 404
    except Exception as e:
        app.logger.error(f"Error serving data.csv: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    
    if ENABLE_DETAILED_LOGGING:
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        if client:
            logger.info("OpenAI integration enabled")
        else:
            logger.warning("OpenAI integration disabled - no API key provided")
    
    app.run(debug=debug, host=host, port=port)