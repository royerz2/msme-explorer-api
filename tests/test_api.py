import pytest
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_data():
    # Create sample data for testing
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Respondent Age': ['25-34', '35-44', '45-54', '25-34', '35-44'],
        'Education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master'],
        'Position': ['Manager', 'Owner', 'Manager', 'Owner', 'Manager'],
        'City': ['City1', 'City2', 'City1', 'City2', 'City1'],
        'Business fields': ['Tech', 'Retail', 'Tech', 'Retail', 'Tech'],
        'AU': [4, 3, 5, 4, 3],
        'INN': [3, 4, 3, 5, 4],
        'RT': [4, 3, 4, 4, 3],
        'IT_SM': [3, 4, 3, 5, 4],
        'IT_CS': [4, 3, 4, 4, 3],
        'Double partnership (DP)': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Triple partnership (TP)': ['No', 'Yes', 'No', 'Yes', 'No']
    }
    return pd.DataFrame(data)

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert 'total_records' in data
    assert data['status'] == 'healthy'

def test_demographics(client):
    """Test the demographics endpoint"""
    response = client.get('/api/demographics')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'distributions' in data
    assert 'cross_tabulations' in data
    
    # Check if all demographic columns are present
    demographics = data['distributions']
    assert 'gender' in demographics
    assert 'respondent_age' in demographics
    assert 'education' in demographics

def test_survey_analysis(client):
    """Test the survey analysis endpoint"""
    response = client.get('/api/survey-analysis')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'basic_statistics' in data
    assert 'correlation_matrix' in data
    assert 'top_correlations' in data
    
    # Check basic statistics
    stats = data['basic_statistics']
    assert 'AU' in stats
    assert 'INN' in stats
    assert all(key in stats['AU'] for key in ['mean', 'median', 'std', 'min', 'max'])

def test_comparative_analysis(client):
    """Test the comparative analysis endpoint"""
    response = client.get('/api/comparative-analysis')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'gender' in data
    assert 'age_groups' in data
    assert 'business_fields' in data
    
    # Check gender comparison
    gender_data = data['gender']
    assert 'AU' in gender_data
    assert all(key in gender_data['AU'] for key in ['male_mean', 'female_mean', 't_statistic', 'p_value'])

def test_clustering(client):
    """Test the clustering endpoint"""
    response = client.get('/api/clustering')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'clustering_results' in data
    assert 'visualization' in data
    
    # Check clustering results
    results = data['clustering_results']
    assert 'k_3' in results
    assert 'k_4' in results
    assert 'k_5' in results

def test_technology_analysis(client):
    """Test the technology analysis endpoint"""
    response = client.get('/api/technology-analysis')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'technology_statistics' in data
    assert 'technology_by_demographics' in data
    assert 'technology_performance_correlation' in data
    
    # Check technology statistics
    tech_stats = data['technology_statistics']
    assert 'IT_SM' in tech_stats
    assert 'IT_CS' in tech_stats

def test_partnership_analysis(client):
    """Test the partnership analysis endpoint"""
    response = client.get('/api/partnership-analysis')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'partnership_distribution' in data
    assert 'partnership_impact' in data
    
    # Check partnership distribution
    dist = data['partnership_distribution']
    assert 'double_partnership_dp' in dist
    assert 'triple_partnership_tp' in dist

def test_comprehensive_report(client):
    """Test the comprehensive report endpoint"""
    response = client.get('/api/comprehensive-report')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert 'sample_info' in data
    assert 'key_findings' in data
    
    # Check sample info
    sample_info = data['sample_info']
    assert 'total_respondents' in sample_info
    assert 'complete_responses' in sample_info
    assert 'survey_variables' in sample_info

def test_error_handling(client):
    """Test error handling for invalid endpoints"""
    response = client.get('/api/invalid-endpoint')
    assert response.status_code == 404

def test_data_validation(client):
    """Test data validation in responses"""
    response = client.get('/api/survey-analysis')
    data = json.loads(response.data)
    
    # Check if all numeric values are valid
    stats = data['basic_statistics']
    for var in stats:
        assert not np.isnan(stats[var]['mean'])
        assert not np.isnan(stats[var]['std'])
        assert not np.isnan(stats[var]['min'])
        assert not np.isnan(stats[var]['max'])

def test_correlation_validation(client):
    """Test correlation matrix validation"""
    response = client.get('/api/survey-analysis')
    data = json.loads(response.data)
    
    corr_matrix = data['correlation_matrix']
    assert 'variables' in corr_matrix
    assert 'matrix' in corr_matrix
    
    # Check if correlation matrix is symmetric
    matrix = np.array(corr_matrix['matrix'])
    assert np.allclose(matrix, matrix.T)
    
    # Check if diagonal elements are 1
    assert np.allclose(np.diag(matrix), 1.0) 