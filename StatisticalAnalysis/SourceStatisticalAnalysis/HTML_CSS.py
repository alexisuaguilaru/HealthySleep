from os import getenv
from datetime import datetime

CurrentYear = datetime.today().year
Home_URL = getenv('DOMAIN','localhost:8080')
StatisticalAnalysis_URL = getenv('STATISTICAL_ANALYSIS_URL','localhost:5050')
DataMining_URL = getenv('DATA_MINING_URL','localhost:5151')

Style = """
<head>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</head>

<style>
.nav-links {
    width: 100%;
    margin-top: 50px;
    padding: 20px 0;
    border-top: 1px solid #e0e0e0;
    text-align: center;
    color: #555;
    font-size: 20px;
    font-family: "Times New Roman", Times, serif;;
}

.app-footer {
    width: 100%;
    margin-top: 50px;
    padding: 20px 0;
    border-top: 1px solid #e0e0e0;
    text-align: center;
    color: #555;
    font-size: 16px;
    font-family: "Times New Roman", Times, serif;;
}

.footer-section {
    margin-bottom: 10px;
}

a {
    color: #007bff;
    text-decoration: none;
    margin: 0 10px;
    transition: color 0.3s;
    font-weight: 600;
}

a:hover {
    color: #0056b3;
    text-decoration: underline;
}

.fab {
    margin-right: 5px;
}
</style>
"""

HeaderNav = f"""
<nav>
    <div class="nav-links">
        <a href="{Home_URL}" target="_blank" title="Home Page">
            <i class="fa fa-home"></i> Home Page
        </a>
        |
        <a href="{StatisticalAnalysis_URL}" target="_blank" title="Statistical Analysis Notebook">
            <i class="fa fa-bar-chart"></i> Statistical Analysis
        </a>
        |
        <a href="{DataMining_URL}" target="_blank" title="Data Mining Notebook">
            <i class="fa fa-database"></i> Data Mining
        </a>
    </div>
</nav>

{Style}
"""

Footer = f"""
<div class="app-footer">
    <div class="footer-section">
        <p>© {CurrentYear} Sleep Quality Analysis</p>
    </div>
    
    <div class="footer-links">
        <a href="https://github.com/alexisuaguilaru/HealthySleep" target="_blank" title="GitHub Repository">
            <i class="fa fa-github"></i> GitHub
        </a>
        |
        <a href="https://www.linkedin.com/in/alexis-aguilar-uribe/" target="_blank" title="LinkedIn Profile">
            <i class="fa fa-linkedin"></i> LinkedIn
        </a>
    </div>
</div>

{Style}
"""