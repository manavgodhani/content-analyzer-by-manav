import re
import streamlit as st
import nltk
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textstat import flesch_reading_ease
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import numpy as np
import json

# Sample HTML content for testing
SAMPLE_HTML_OPTIONS = {
    "E-commerce Product Page": """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Premium Wireless Headphones - TechStore</title>
        <script>
            // Product data for dynamic content
            const productData = {
                name: "Premium Wireless Headphones XR-500",
                features: ["Active Noise Cancellation", "30-hour battery", "Bluetooth 5.2"],
                price: 199.99,
                discount: 20
            };
            
            // SEO-relevant content in JavaScript
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('dynamic-content').innerHTML = 
                    '<p>Advanced noise cancellation technology with premium sound quality. ' +
                    'Perfect for music lovers and professionals who demand the best audio experience.</p>';
            });
        </script>
    </head>
    <body>
        <header>
            <nav>
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/products">Products</a></li>
                    <li><a href="/about">About</a></li>
                    <li><a href="/contact">Contact</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <article>
                <h1>Premium Wireless Headphones - Model XR-500</h1>
                <p>Experience crystal-clear audio with our flagship wireless headphones. Featuring advanced noise cancellation technology, 30-hour battery life, and premium comfort design.</p>

                <div id="dynamic-content"></div>

                <h2>Key Features</h2>
                <ul>
                    <li>Active Noise Cancellation (ANC) technology</li>
                    <li>30-hour battery life with quick charge</li>
                    <li>Premium leather ear cushions</li>
                    <li>Bluetooth 5.2 connectivity</li>
                    <li>Touch controls and voice assistant support</li>
                </ul>

                <h2>Product Specifications</h2>
                <p>Our XR-500 headphones are engineered for audiophiles who demand the best. The 40mm dynamic drivers deliver rich, detailed sound across all frequencies. The adaptive noise cancellation automatically adjusts to your environment, whether you're on a busy street or in a quiet office.</p>

                <p>The headphones feature premium materials including aircraft-grade aluminum and genuine leather. The adjustable headband ensures comfortable wear for extended listening sessions. Compatible with all major devices including smartphones, tablets, and laptops.</p>

                <h2>Customer Reviews</h2>
                <p>Rated 4.8/5 stars by over 2,500 customers. Users consistently praise the sound quality, comfort, and battery life. Many note these headphones rival products costing twice as much.</p>

                <h2>Pricing and Availability</h2>
                <p>Available now for $199.99 with free shipping. 30-day money-back guarantee and 2-year warranty included. Limited time offer: save 20% with code AUDIO20.</p>
            </article>
        </main>

        <aside>
            <h3>Related Products</h3>
            <ul>
                <li>Wireless Earbuds Pro - $129</li>
                <li>Gaming Headset Elite - $159</li>
                <li>Portable Speaker Max - $89</li>
            </ul>
        </aside>

        <footer>
            <p>&copy; 2024 TechStore. All rights reserved. Privacy Policy | Terms of Service</p>
        </footer>
    </body>
    </html>
    """,

    "Blog Article": """
    <!DOCTYPE html>
    <html>
    <head>
        <title>10 SEO Tips for Better Rankings in 2024</title>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "headline": "10 SEO Tips for Better Rankings in 2024",
            "description": "Essential SEO strategies for improving search rankings in 2024",
            "author": {
                "@type": "Person",
                "name": "SEO Expert"
            }
        }
        </script>
    </head>
    <body>
        <header>
            <nav>
                <a href="/">Home</a>
                <a href="/blog">Blog</a>
                <a href="/services">Services</a>
                <a href="/contact">Contact</a>
            </nav>
        </header>

        <main>
            <article>
                <h1>10 SEO Tips for Better Rankings in 2024</h1>
                <p>Search engine optimization continues to evolve rapidly. Here are the most effective strategies to improve your website's visibility in search results this year.</p>

                <h2>1. Focus on User Experience</h2>
                <p>Google's Core Web Vitals have become crucial ranking factors. Optimize your page load speed, interactivity, and visual stability. Users expect fast, responsive websites that work seamlessly across all devices.</p>

                <h2>2. Create High-Quality Content</h2>
                <p>Content remains king in SEO. Focus on creating comprehensive, well-researched articles that genuinely help your audience. Use natural language and answer the questions your users are actually asking.</p>

                <h2>3. Optimize for Mobile-First</h2>
                <p>With mobile-first indexing, Google primarily uses the mobile version of your site for ranking. Ensure your mobile experience is exceptional with responsive design and fast loading times.</p>

                <h2>4. Build Quality Backlinks</h2>
                <p>Earn links from authoritative websites in your industry. Focus on creating linkable assets like original research, comprehensive guides, or useful tools that others naturally want to reference.</p>

                <h2>5. Use Schema Markup</h2>
                <p>Structured data helps search engines understand your content better. Implement relevant schema markup to enhance your search listings with rich snippets, ratings, and other enhanced features.</p>

                <p>These strategies form the foundation of effective SEO in 2024. Remember that SEO is a long-term investment that requires consistent effort and adaptation to algorithm changes.</p>
            </article>
        </main>

        <aside>
            <h3>Popular Articles</h3>
            <ul>
                <li>Content Marketing Strategy Guide</li>
                <li>Technical SEO Checklist</li>
                <li>Local SEO Best Practices</li>
            </ul>
        </aside>

        <footer>
            <p>Published by SEO Experts | Follow us on social media</p>
        </footer>
    </body>
    </html>
    """,

    "Soft 404 Example": """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page Not Found - Example Site</title>
    </head>
    <body>
        <header>
            <nav>
                <a href="/">Home</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
            </nav>
        </header>

        <main>
            <h1>Oops! Something went wrong</h1>
            <p>Sorry, this page does not exist or has been moved.</p>
            <p>The page you're looking for cannot be found. Please check the URL or return to our homepage.</p>
        </main>

        <footer>
            <p>&copy; 2024 Example Site</p>
        </footer>
    </body>
    </html>
    """
}

def load_css():
    """Enhanced CSS with better theme compatibility and sidebar borders"""
    st.markdown("""
    <style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --light-bg: rgba(255, 255, 255, 0.05);
        --border-color: rgba(255, 255, 255, 0.1);
    }

    /* Sidebar styling with borders */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-right: 3px solid var(--primary-color);
        box-shadow: 2px 0 10px rgba(102, 126, 234, 0.2);
    }

    .css-1d391kg .stMarkdown {
        color: white !important;
    }

    .css-1d391kg h3 {
        color: white !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .css-1d391kg .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .css-1d391kg .stCheckbox > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Main app header */
    .app-header {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: 15px;
        color: white !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }

    .app-header h1 {
        font-size: 2.5rem;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
        color: white !important;
    }

    .app-header p {
        font-size: 1.1rem;
        margin: 0;
        opacity: 0.95;
        color: white !important;
    }

    /* Metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }

    .metric-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        background: var(--light-bg);
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        display: block;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        font-weight: 500;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid transparent;
    }

    .status-healthy { background: rgba(40, 167, 69, 0.2); color: #28a745; border-color: rgba(40, 167, 69, 0.3); }
    .status-warning { background: rgba(255, 193, 7, 0.2); color: #e67e22; border-color: rgba(255, 193, 7, 0.3); }
    .status-error { background: rgba(220, 53, 69, 0.2); color: #dc3545; border-color: rgba(220, 53, 69, 0.3); }

    /* Content areas */
    .content-area {
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        background: var(--light-bg);
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .area-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    .area-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
        opacity: 0.9;
    }

    .weight-badge {
        background: var(--primary-color);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .content-preview {
        font-size: 0.9rem;
        opacity: 0.7;
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    /* Recommendations */
    .recommendation {
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }

    .rec-success { background: rgba(40, 167, 69, 0.15); border-left-color: var(--success-color); color: #28a745; }
    .rec-warning { background: rgba(255, 193, 7, 0.15); border-left-color: var(--warning-color); color: #e67e22; }
    .rec-danger { background: rgba(220, 53, 69, 0.15); border-left-color: var(--danger-color); color: #dc3545; }

    /* Sample selector */
    .sample-selector {
        background: var(--light-bg);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }

    /* Chart containers */
    .chart-section {
        margin: 2rem 0;
        padding: 1rem;
        background: var(--light-bg);
        border-radius: 10px;
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }

    /* Theme compatibility */
    [data-theme="dark"] .metric-card,
    [data-theme="dark"] .content-area,
    [data-theme="dark"] .chart-section,
    [data-theme="dark"] .sample-selector {
        --light-bg: rgba(255, 255, 255, 0.05);
        --border-color: rgba(255, 255, 255, 0.1);
    }

    [data-theme="light"] .metric-card,
    [data-theme="light"] .content-area,
    [data-theme="light"] .chart-section,
    [data-theme="light"] .sample-selector {
        --light-bg: rgba(255, 255, 255, 0.8);
        --border-color: rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def download_nltk_data():
    """Download NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("📥 Downloading language models..."):
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

def extract_js_content(soup):
    """Extract meaningful content from JavaScript, including JSON-LD and data"""
    js_content = []
    
    # Extract JSON-LD structured data
    json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
    for script in json_ld_scripts:
        try:
            data = json.loads(script.string or script.get_text())
            # Extract text content from structured data
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        js_content.append(value)
                    elif isinstance(value, dict) and 'name' in value:
                        js_content.append(str(value.get('name', '')))
        except (json.JSONDecodeError, AttributeError):
            continue
    
    # Extract content from regular script tags
    script_tags = soup.find_all('script', {'type': lambda x: x != 'application/ld+json'})
    for script in script_tags:
        script_text = script.string or script.get_text()
        if script_text:
            # Extract string literals that look like content
            string_matches = re.findall(r'["\']([^"\']{20,})["\']', script_text)
            for match in string_matches:
                # Filter out code-like strings and URLs
                if not re.search(r'[{}();=<>]|https?://', match) and len(match.split()) > 3:
                    js_content.append(match)
            
            # Extract object properties that contain content
            prop_matches = re.findall(r'(?:name|title|description|content|text)\s*[:=]\s*["\']([^"\']{10,})["\']', script_text, re.IGNORECASE)
            js_content.extend(prop_matches)
    
    return ' '.join(js_content) if js_content else ''

def analyze_html_structure(html_content):
    """Enhanced HTML structure analysis with better content extraction and JS support"""
    soup = BeautifulSoup(html_content, 'html.parser')

    content_areas = {
        'main_content': {'weight': 10, 'content': '', 'icon': '🎯'},
        'header': {'weight': 3, 'content': '', 'icon': '📋'},
        'nav': {'weight': 2, 'content': '', 'icon': '🧭'},
        'sidebar': {'weight': 2, 'content': '', 'icon': '📄'},
        'footer': {'weight': 1, 'content': '', 'icon': '📎'},
        'javascript': {'weight': 1, 'content': '', 'icon': '⚡'}  # New JS content area
    }

    # Extract JavaScript content first
    js_content = extract_js_content(soup)
    if js_content:
        content_areas['javascript']['content'] = js_content

    # Enhanced main content selectors with better prioritization
    main_selectors = [
        # Semantic HTML5 elements (highest priority)
        'main', 'article', '[role="main"]',
        # Content-specific classes (high priority)
        '.main-content', '#main', '.content', '.page-content', '.main-wrapper',
        '.content-wrapper', '.post-content', '.entry-content', '.article-content',
        # Domain-specific selectors (medium priority)  
        '.product-details', '.loan-details', '.banking-content', '.loan-info',
        '.product-info', '.features', '.benefits', '.eligibility', '.documentation',
        '.details-section', '.loan-features', '.product-highlights', '.section-content',
        # Generic layout selectors (lower priority)
        '.container .content', '.page-container .content', '.row .col',
        '.container .row', '.grid-content',
        # Fallback selectors (lowest priority)
        '[class*="content"]', '[class*="main"]', '[id*="content"]',
        '[class*="section"]', '[class*="details"]', '[id*="main"]'
    ]
    
    # Try selectors in priority order
    for selector in main_selectors:
        try:
            elements = soup.select(selector)
            if elements:
                # Filter out navigation, footer, sidebar elements
                filtered_elements = []
                for elem in elements:
                    elem_text = elem.get_text(strip=True).lower()
                    elem_classes = ' '.join(elem.get('class', [])).lower()
                    elem_id = (elem.get('id') or '').lower()
                    
                    # Skip obvious non-content elements
                    if any(skip_term in elem_classes + ' ' + elem_id for skip_term in 
                           ['nav', 'menu', 'footer', 'sidebar', 'banner', 'ad', 'advertisement']):
                        continue
                        
                    filtered_elements.append(elem)
                
                if filtered_elements:
                    content_text = ' '.join([elem.get_text(strip=True) for elem in filtered_elements])
                    # Only use if we get substantial content
                    if len(content_text.split()) > 15:
                        content_areas['main_content']['content'] = content_text
                        break
        except Exception:
            continue

    # Enhanced fallback strategy
    if not content_areas['main_content']['content'] or len(content_areas['main_content']['content'].split()) < 30:
        # Strategy 1: Find content-heavy sections
        all_sections = soup.find_all(['section', 'div', 'article', 'main', 'span'], class_=True)
        content_candidates = []
        
        for section in all_sections:
            try:
                section_classes = ' '.join(section.get('class', [])).lower()
                section_id = (section.get('id') or '').lower()
                
                # Skip unwanted elements
                if any(word in section_classes + ' ' + section_id for word in 
                       ['nav', 'menu', 'footer', 'sidebar', 'banner', 'ad', 'cookie', 'popup', 'modal']):
                    continue
                    
                text = section.get_text(strip=True)
                word_count = len(text.split())
                
                # Look for substantial content sections
                if word_count > 20:
                    # Boost score for content-indicating classes
                    score_multiplier = 1
                    if any(term in section_classes for term in 
                           ['content', 'main', 'article', 'post', 'product', 'detail', 'info', 'section']):
                        score_multiplier = 2.5
                    elif any(term in section_classes for term in
                             ['feature', 'benefit', 'description', 'overview', 'summary']):
                        score_multiplier = 2
                        
                    content_candidates.append((section, word_count * score_multiplier, text))
            except Exception:
                continue
        
        # Combine top content sections
        if content_candidates:
            content_candidates.sort(key=lambda x: x[1], reverse=True)
            
            combined_content = []
            total_words = 0
            
            for i, (section, score, text) in enumerate(content_candidates[:5]):  # Top 5 sections
                if len(text.split()) > 15 and total_words < 2000:  # Limit total content
                    combined_content.append(text)
                    total_words += len(text.split())
            
            if combined_content:
                best_content = ' '.join(combined_content)
                current_words = len(content_areas['main_content']['content'].split()) if content_areas['main_content']['content'] else 0
                if len(best_content.split()) > max(current_words, 20):
                    content_areas['main_content']['content'] = best_content

    # Strategy 2: Clean body extraction if still insufficient
    if not content_areas['main_content']['content'] or len(content_areas['main_content']['content'].split()) < 15:
        body = soup.find('body')
        if body:
            # Remove unwanted elements
            for unwanted in body.find_all(['script', 'style', 'noscript', 'iframe', 'svg']):
                unwanted.decompose()
            
            # Remove elements with unwanted classes/IDs
            for tag in body.find_all(attrs={'class': True}):
                class_names = ' '.join(tag.get('class', [])).lower()
                if any(word in class_names for word in 
                       ['header', 'footer', 'nav', 'menu', 'sidebar', 'ad', 'banner', 'cookie']):
                    tag.decompose()
            
            content_areas['main_content']['content'] = body.get_text(strip=True, separator=' ')

    # Extract other content areas with improved selectors
    # Header content
    header_selectors = ['header', '.header', '#header', '[role="banner"]']
    for selector in header_selectors:
        elements = soup.select(selector)
        if elements:
            content_areas['header']['content'] = ' '.join([elem.get_text(strip=True) for elem in elements])
            break

    # Navigation content  
    nav_selectors = ['nav', '.nav', '.navigation', '#nav', '[role="navigation"]']
    nav_content = []
    for selector in nav_selectors:
        elements = soup.select(selector)
        nav_content.extend([elem.get_text(strip=True) for elem in elements])
    if nav_content:
        content_areas['nav']['content'] = ' '.join(nav_content)

    # Footer content
    footer_selectors = ['footer', '.footer', '#footer', '[role="contentinfo"]']
    for selector in footer_selectors:
        elements = soup.select(selector)
        if elements:
            content_areas['footer']['content'] = ' '.join([elem.get_text(strip=True) for elem in elements])
            break

    # Sidebar content
    sidebar_selectors = [
        'aside', '.sidebar', '.aside', '#sidebar', '[role="complementary"]',
        '.side-content', '.secondary', '.widget-area', '.right-sidebar', '.left-sidebar'
    ]
    for selector in sidebar_selectors:
        elements = soup.select(selector)
        if elements:
            content_areas['sidebar']['content'] = ' '.join([elem.get_text(strip=True) for elem in elements])
            break

    return content_areas

def tokenize_content(text):
    """Optimized tokenization with better filtering"""
    if not text or not text.strip():
        return []

    # Clean text first
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        # Enhanced filtering
        tokens = [token for token in tokens 
                 if token.isalpha() and len(token) > 2 and token not in stop_words]
        
        return tokens
    except Exception:
        # Fallback to simple split
        return [word for word in text.split() 
                if len(word) > 2 and word.isalpha()]

def calculate_importance_scores(content_areas):
    """Calculate token importance with improved scoring"""
    scored_tokens = {}
    total_weight = 0

    for area, data in content_areas.items():
        if not data['content'] or not data['content'].strip():
            continue

        tokens = tokenize_content(data['content'])
        if not tokens:
            continue
            
        weight = data['weight']
        token_count = Counter(tokens)

        for token, freq in token_count.items():
            # Apply frequency and position weighting
            score = weight * min(freq, 5)  # Cap frequency impact
            
            if token in scored_tokens:
                scored_tokens[token] += score
            else:
                scored_tokens[token] = score
            total_weight += score

    # Normalize scores to percentages
    if total_weight > 0:
        for token in scored_tokens:
            scored_tokens[token] = (scored_tokens[token] / total_weight) * 100

    return scored_tokens

def detect_soft_404(content_areas):
    """Enhanced soft 404 detection"""
    main_content = content_areas['main_content']['content']

    if not main_content or not main_content.strip():
        return {
            'is_soft_404': True,
            'reason': 'No main content found',
            'confidence': 'High',
            'word_count': 0
        }

    word_count = len([word for word in main_content.split() if word.strip()])

    # Enhanced error phrase detection
    error_phrases = [
        'page not found', '404', 'not found', 'page does not exist',
        'sorry, this page', 'oops', 'something went wrong', 'error occurred',
        'page cannot be found', 'broken link', 'page unavailable',
        'content not available', 'resource not found', 'invalid url',
        'page has been removed', 'no longer exists', 'temporarily unavailable'
    ]

    main_lower = main_content.lower()
    error_count = sum(1 for phrase in error_phrases if phrase in main_lower)
    
    if error_count > 0:
        confidence = 'High' if error_count > 1 or word_count < 100 else 'Medium'
        return {
            'is_soft_404': True,
            'reason': f'Contains {error_count} error phrase(s)',
            'confidence': confidence,
            'word_count': word_count
        }

    # Content quality thresholds
    if word_count < 30:
        return {
            'is_soft_404': True,
            'reason': f'Extremely thin content ({word_count} words)',
            'confidence': 'High',
            'word_count': word_count
        }
    elif word_count < 50:
        return {
            'is_soft_404': True,
            'reason': f'Very thin content ({word_count} words)',
            'confidence': 'Medium',
            'word_count': word_count
        }

    return {
        'is_soft_404': False,
        'reason': 'Content appears healthy',
        'confidence': 'Low',
        'word_count': word_count
    }

def analyze_readability(content_areas):
    """Enhanced readability analysis"""
    main_content = content_areas['main_content']['content']
    if not main_content or len(main_content.strip()) < 10:
        return {'flesch_score': 0, 'grade_level': 'N/A', 'readability': 'Cannot analyze'}

    try:
        flesch_score = flesch_reading_ease(main_content)
        
        # Readability mapping
        if flesch_score >= 90:
            grade_level, readability = "5th grade", "Very Easy"
        elif flesch_score >= 80:
            grade_level, readability = "6th grade", "Easy"
        elif flesch_score >= 70:
            grade_level, readability = "7th grade", "Fairly Easy"
        elif flesch_score >= 60:
            grade_level, readability = "8th-9th grade", "Standard"
        elif flesch_score >= 50:
            grade_level, readability = "10th-12th grade", "Fairly Difficult"
        elif flesch_score >= 30:
            grade_level, readability = "College level", "Difficult"
        else:
            grade_level, readability = "Graduate level", "Very Difficult"

        return {
            'flesch_score': round(flesch_score, 1),
            'grade_level': grade_level,
            'readability': readability
        }
    except Exception:
        return {'flesch_score': 0, 'grade_level': 'N/A', 'readability': 'Cannot analyze'}

def create_app_header():
    """Create the main app header"""
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Google Content Analyzer</h1>
        <p>Analyze content structure like Google does - Based on Gary Illyes' methodology</p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_dashboard(content_areas, soft_404_result, importance_scores, readability_data):
    """Create enhanced metrics dashboard"""
    main_word_count = soft_404_result['word_count']
    total_areas = sum(1 for area, data in content_areas.items() if data['content'].strip())
    unique_tokens = len(importance_scores)
    flesch_score = readability_data['flesch_score']
    js_words = len(content_areas['javascript']['content'].split()) if content_areas['javascript']['content'] else 0

    # Status determination
    if soft_404_result['is_soft_404']:
        status_class, status_text = "status-error", "⚠️ Soft 404 Risk"
    else:
        status_class, status_text = "status-healthy", "✅ Healthy Page"

    # Readability status
    if flesch_score >= 60:
        read_class, read_text = "status-healthy", "✅ Good Readability"
    elif flesch_score >= 30:
        read_class, read_text = "status-warning", "⚠️ Hard to Read"
    else:
        read_class, read_text = "status-error", "🔴 Very Hard to Read"

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <span class="metric-value">{main_word_count}</span>
            <div class="metric-label">Main Content Words</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">{js_words}</span>
            <div class="metric-label">JavaScript Content</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">{total_areas}</span>
            <div class="metric-label">Content Areas Found</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">{unique_tokens}</span>
            <div class="metric-label">Unique Tokens</div>
        </div>
        <div class="metric-card">
            <span class="metric-value">{flesch_score}</span>
            <div class="metric-label">Readability Score</div>
        </div>
        <div class="metric-card">
            <div class="status-badge {status_class}">{status_text}</div>
            <div class="metric-label">Page Status</div>
        </div>
        <div class="metric-card">
            <div class="status-badge {read_class}">{read_text}</div>
            <div class="metric-label">Reading Level</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_importance_chart(importance_scores):
    """Create token importance chart"""
    if not importance_scores:
        st.info("📊 No tokens found for importance analysis")
        return

    top_tokens = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
    if not top_tokens:
        st.warning("No significant tokens found")
        return

    fig = go.Figure(data=[
        go.Bar(
            y=list(top_tokens.keys()),
            x=list(top_tokens.values()),
            orientation='h',
            marker_color='#667eea',
            text=[f'{v:.1f}%' for v in top_tokens.values()],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="🎯 Token Importance Scores (Top 15)",
        xaxis_title="Importance Score (%)",
        yaxis_title="Tokens",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
        showlegend=False,
        margin=dict(l=120, r=50, t=60, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

def create_distribution_chart(content_areas):
    """Create content distribution chart"""
    area_data = {}
    for area, data in content_areas.items():
        if data['content'].strip():
            word_count = len([word for word in data['content'].split() if word.strip()])
            if word_count > 0:
                area_data[area] = word_count

    if not area_data:
        st.info("📊 No content distribution data available")
        return

    labels = [area.replace('_', ' ').title() for area in area_data.keys()]
    values = list(area_data.values())
    colors = ['#667eea', '#764ba2', '#f093fb', '#fad0c4', '#a8edea', '#ff9a9e']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors[:len(labels)]
    )])

    fig.update_layout(
        title="📊 Content Distribution by Area",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333')
    )

    st.plotly_chart(fig, use_container_width=True)

def create_weight_impact_chart(content_areas, importance_scores):
    """Create weighted impact visualization"""
    if not importance_scores:
        st.info("📊 No data for weight impact analysis")
        return

    # Calculate tokens per area with their weights
    area_tokens = {}
    for area, data in content_areas.items():
        if data['content'].strip():
            tokens = tokenize_content(data['content'])
            unique_tokens = len(set(tokens))
            if unique_tokens > 0:
                area_tokens[area.replace('_', ' ').title()] = {
                    'tokens': unique_tokens,
                    'weight': data['weight'],
                    'weighted_impact': unique_tokens * data['weight']
                }

    if not area_tokens:
        st.info("📊 No token data available for weight analysis")
        return

    areas = list(area_tokens.keys())
    tokens = [area_tokens[area]['tokens'] for area in areas]
    weights = [area_tokens[area]['weight'] for area in areas]
    impacts = [area_tokens[area]['weighted_impact'] for area in areas]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Token Count vs Weight", "Weighted Impact"),
        specs=[[{"secondary_y": True}, {"type": "bar"}]]
    )

    # Left chart: Token count vs weight
    fig.add_trace(
        go.Bar(x=areas, y=tokens, name="Unique Tokens", marker_color='#667eea'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=areas, y=weights, mode='lines+markers', name="Weight Multiplier", 
                  line=dict(color='#e74c3c', width=3), marker=dict(size=8)),
        row=1, col=1, secondary_y=True
    )

    # Right chart: Weighted impact
    fig.add_trace(
        go.Bar(x=areas, y=impacts, name="Weighted Impact", marker_color='#2ecc71'),
        row=1, col=2
    )

    fig.update_layout(
        title="⚖️ Content Area Weight Impact Analysis",
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(title_text="Content Areas", row=1, col=1)
    fig.update_xaxes(title_text="Content Areas", row=1, col=2)
    fig.update_yaxes(title_text="Token Count", row=1, col=1)
    fig.update_yaxes(title_text="Weight Multiplier", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Weighted Impact Score", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

def create_token_frequency_chart(importance_scores):
    """Create token frequency distribution chart"""
    if not importance_scores:
        st.info("📊 No frequency data available")
        return

    # Group tokens by importance ranges
    ranges = {
        'High Impact (4%+)': 0,
        'Medium Impact (2-4%)': 0,
        'Low Impact (1-2%)': 0,
        'Minimal Impact (<1%)': 0
    }

    for token, score in importance_scores.items():
        if score >= 4:
            ranges['High Impact (4%+)'] += 1
        elif score >= 2:
            ranges['Medium Impact (2-4%)'] += 1
        elif score >= 1:
            ranges['Low Impact (1-2%)'] += 1
        else:
            ranges['Minimal Impact (<1%)'] += 1

    fig = go.Figure(data=[
        go.Bar(
            x=list(ranges.keys()),
            y=list(ranges.values()),
            marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6'],
            text=list(ranges.values()),
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="📈 Token Impact Distribution",
        xaxis_title="Impact Categories",
        yaxis_title="Number of Tokens",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333')
    )

    st.plotly_chart(fig, use_container_width=True)

def create_content_quality_radar(content_areas, soft_404_result, importance_scores, readability_data):
    """Create content quality radar chart"""
    # Calculate quality metrics (0-100 scale)
    metrics = {}

    # Content volume score
    word_count = soft_404_result['word_count']
    if word_count >= 1000:
        metrics['Content Volume'] = 100
    elif word_count >= 500:
        metrics['Content Volume'] = 80
    elif word_count >= 300:
        metrics['Content Volume'] = 60
    elif word_count >= 100:
        metrics['Content Volume'] = 40
    else:
        metrics['Content Volume'] = 20

    # Structural completeness
    area_count = sum(1 for area, data in content_areas.items() if data['content'].strip())
    metrics['Structure Completeness'] = min(100, (area_count / 6) * 100)

    # Token diversity
    token_count = len(importance_scores)
    if token_count >= 100:
        metrics['Vocabulary Diversity'] = 100
    elif token_count >= 50:
        metrics['Vocabulary Diversity'] = 80
    elif token_count >= 25:
        metrics['Vocabulary Diversity'] = 60
    elif token_count >= 10:
        metrics['Vocabulary Diversity'] = 40
    else:
        metrics['Vocabulary Diversity'] = 20

    # Readability score (convert Flesch to 0-100)
    flesch_score = readability_data['flesch_score']
    if flesch_score >= 60:
        metrics['Readability'] = min(100, flesch_score)
    else:
        metrics['Readability'] = max(0, flesch_score)

    # SEO Health (inverse of soft 404 risk)
    if soft_404_result['is_soft_404']:
        metrics['SEO Health'] = 20
    else:
        metrics['SEO Health'] = 100

    # Content balance (how well distributed content is)
    if len([area for area, data in content_areas.items() if data['content'].strip()]) >= 3:
        metrics['Content Balance'] = 100
    else:
        metrics['Content Balance'] = 50

    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Content Quality',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="🎯 Content Quality Radar",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def create_seo_score_gauge(content_areas, soft_404_result, importance_scores, readability_data):
    """Create overall SEO score gauge"""
    # Calculate component scores
    word_count = soft_404_result['word_count']
    area_count = sum(1 for area, data in content_areas.items() if data['content'].strip())
    token_count = len(importance_scores)
    flesch_score = readability_data['flesch_score']

    # Component scoring (0-100)
    scores = {}
    scores['content'] = min(100, max(20, (word_count / 500) * 100)) if word_count >= 50 else 20
    scores['structure'] = min(100, (area_count / 6) * 100)
    scores['diversity'] = min(100, max(40, (token_count / 50) * 100)) if token_count >= 10 else 40
    scores['readability'] = max(0, min(100, flesch_score)) if flesch_score >= 30 else 40
    scores['soft_404'] = 0 if soft_404_result['is_soft_404'] else 100
    
    # JavaScript content bonus
    js_bonus = 5 if content_areas['javascript']['content'].strip() else 0

    # Weighted average with bonus
    overall_score = (
        scores['content'] * 0.30 +
        scores['structure'] * 0.25 +
        scores['diversity'] * 0.20 +
        scores['readability'] * 0.15 +
        scores['soft_404'] * 0.10 +
        js_bonus
    )

    # Determine status
    if overall_score >= 80:
        color, status = "#2ecc71", "Excellent"
    elif overall_score >= 60:
        color, status = "#f39c12", "Good"
    elif overall_score >= 40:
        color, status = "#e67e22", "Needs Work"
    else:
        color, status = "#e74c3c", "Poor"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"SEO Score: {status}"},
        delta={'reference': 75},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 80], 'color': "lightblue"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))

    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"}, paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Show score breakdown
    with st.expander("📊 Score Breakdown"):
        breakdown_data = {
            'Component': ['Content Volume (30%)', 'Structure (25%)', 'Vocabulary (20%)', 
                         'Readability (15%)', 'No Soft 404 (10%)'],
            'Score': [f"{scores['content']:.0f}/100", f"{scores['structure']:.0f}/100", 
                     f"{scores['diversity']:.0f}/100", f"{scores['readability']:.0f}/100", 
                     f"{scores['soft_404']:.0f}/100"],
            'Weight': ['30%', '25%', '20%', '15%', '10%'],
            'Contribution': [f"{scores['content'] * 0.30:.1f}", f"{scores['structure'] * 0.25:.1f}",
                           f"{scores['diversity'] * 0.20:.1f}", f"{scores['readability'] * 0.15:.1f}",
                           f"{scores['soft_404'] * 0.10:.1f}"]
        }
        df = pd.DataFrame(breakdown_data)
        st.dataframe(df, use_container_width=True)

def create_content_areas_display(content_areas):
    """Display content areas in cards"""
    st.subheader("📋 Content Areas Analysis")

    for area, data in content_areas.items():
        if data['content'].strip():
            word_count = len([word for word in data['content'].split() if word.strip()])
            preview = data['content'][:300] + '...' if len(data['content']) > 300 else data['content']

            st.markdown(f"""
            <div class="content-area">
                <div class="area-header">
                    <h3 class="area-title">{data['icon']} {area.replace('_', ' ').title()}</h3>
                    <span class="weight-badge">Weight: {data['weight']}x</span>
                </div>
                <p><strong>Words:</strong> {word_count} | <strong>Characters:</strong> {len(data['content'])}</p>
                <div class="content-preview">{preview}</div>
            </div>
            """, unsafe_allow_html=True)

def create_recommendations(content_areas, soft_404_result, importance_scores, readability_data):
    """Generate comprehensive SEO recommendations"""
    st.subheader("💡 SEO Recommendations")

    recommendations = []

    # Soft 404 analysis
    if soft_404_result['is_soft_404']:
        recommendations.append(("danger", f"🚨 Soft 404 Detected: {soft_404_result['reason']}"))
        recommendations.append(("warning", "📝 Add substantial, unique content to fix soft 404 issues"))
    else:
        recommendations.append(("success", "✅ Page appears healthy - no soft 404 issues detected"))

    # Content analysis
    word_count = soft_404_result['word_count']
    if word_count < 300:
        recommendations.append(("warning", f"⚠️ Thin Content: {word_count} words. Aim for 300+ words for better SEO."))
    else:
        recommendations.append(("success", f"✅ Good Content Length: {word_count} words"))

    # JavaScript content analysis
    js_content = content_areas['javascript']['content']
    if js_content:
        js_words = len(js_content.split())
        recommendations.append(("success", f"✅ JavaScript Content Found: {js_words} words from structured data and scripts"))
    else:
        recommendations.append(("warning", "⚠️ No JavaScript content detected. Consider adding structured data (JSON-LD)"))

    # Structure analysis
    if not content_areas['main_content']['content']:
        recommendations.append(("danger", "🔴 No Main Content: Use semantic HTML elements like <main> or <article>"))

    # Token diversity
    token_count = len(importance_scores)
    if token_count > 50:
        recommendations.append(("success", f"✅ Rich Vocabulary: {token_count} unique tokens"))
    elif token_count < 20:
        recommendations.append(("warning", "⚠️ Limited Vocabulary: Consider more diverse terminology"))

    # Readability analysis
    flesch_score = readability_data['flesch_score']
    if flesch_score < 30:
        recommendations.append(("warning", f"⚠️ Difficult to Read: Flesch score {flesch_score}. Use shorter sentences and simpler words."))
    elif flesch_score >= 60:
        recommendations.append(("success", f"✅ Good Readability: Flesch score {flesch_score}"))

    # Display recommendations
    for rec_type, message in recommendations:
        st.markdown(f'<div class="recommendation rec-{rec_type}">{message}</div>', unsafe_allow_html=True)

def analyze_url(url, show_advanced_charts=True):
    """Analyze a URL with progress tracking"""
    try:
        progress = st.progress(0)
        status = st.empty()

        status.text("🌐 Fetching webpage...")
        progress.progress(25)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        progress.progress(50)
        status.text("🔍 Analyzing content...")

        analyze_html(response.text, url, show_advanced_charts)

        progress.progress(100)
        status.text("✅ Analysis complete!")
        time.sleep(1)
        progress.empty()
        status.empty()

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching URL: {str(e)}")
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")

def analyze_html(html_content, source, show_advanced_charts=True):
    """Main analysis function with enhanced display"""
    # Perform analysis
    content_areas = analyze_html_structure(html_content)
    importance_scores = calculate_importance_scores(content_areas)
    soft_404_result = detect_soft_404(content_areas)
    readability_data = analyze_readability(content_areas)

    # Display results
    st.markdown("## 📊 Analysis Results")
    st.markdown(f"**Source:** {source}")

    # Main dashboard
    create_metrics_dashboard(content_areas, soft_404_result, importance_scores, readability_data)

    # SEO Score
    if show_advanced_charts:
        st.markdown("### 🎯 Overall SEO Score")
        create_seo_score_gauge(content_areas, soft_404_result, importance_scores, readability_data)

    # Charts section - organized in tabs for better UX
    if show_advanced_charts:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Core Analysis", "⚖️ Weight Impact", "📈 Token Distribution", "🎯 Quality Radar"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="chart-section">', unsafe_allow_html=True)
                create_importance_chart(importance_scores)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="chart-section">', unsafe_allow_html=True)
                create_distribution_chart(content_areas)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            create_weight_impact_chart(content_areas, importance_scores)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            create_token_frequency_chart(importance_scores)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            create_content_quality_radar(content_areas, soft_404_result, importance_scores, readability_data)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Basic charts only
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            create_importance_chart(importance_scores)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            create_distribution_chart(content_areas)
            st.markdown('</div>', unsafe_allow_html=True)

    # Content areas and recommendations
    create_content_areas_display(content_areas)
    create_recommendations(content_areas, soft_404_result, importance_scores, readability_data)

    # Technical insights
    with st.expander("🔬 Technical Insights - Gary Illyes' Methodology"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 How Google Analyzes Content:**

            1. **Positional Analysis**: Google identifies content areas in the rendered HTML
            2. **Weight Assignment**: Different areas get different importance weights
            3. **Token Extraction**: Content is broken into searchable tokens
            4. **Importance Scoring**: Tokens scored based on their location
            5. **Quality Assessment**: Pages checked for thin/error content
            6. **JavaScript Processing**: Extract structured data and dynamic content
            """)

        with col2:
            st.markdown("""
            **📈 Weighting System Used:**

            - **Main Content**: 10x weight (highest priority)
            - **Headers**: 3x weight 
            - **Navigation**: 2x weight
            - **Sidebar**: 2x weight
            - **Footer**: 1x weight
            - **JavaScript**: 1x weight (structured data, dynamic content)
            """)

        st.markdown("---")
        st.markdown(f"""
        **📊 Analysis Summary:**

        - **Total Words**: {soft_404_result['word_count']} in main content
        - **JS Content**: {len(content_areas['javascript']['content'].split())} words from scripts
        - **Readability**: {readability_data['readability']} ({readability_data['grade_level']})
        - **Unique Tokens**: {len(importance_scores)} identified
        - **Content Areas**: {sum(1 for area, data in content_areas.items() if data['content'].strip())} found
        - **Soft 404 Risk**: {'Yes' if soft_404_result['is_soft_404'] else 'No'}
        """)

        st.markdown("---")
        st.markdown("""
        **💡 Key Takeaways from Gary Illyes:**

        - **Focus on Main Content**: Use semantic HTML like `<main>` and `<article>` tags
        - **Avoid Soft 404s**: They're critical errors that waste Google's crawl budget
        - **Content Position Matters**: Where content appears affects its ranking weight
        - **Use Proper Structure**: Help Google understand your page hierarchy
        - **Quality Over Quantity**: Focus on substantial, helpful content in main areas
        - **JavaScript Content**: Modern sites should include structured data (JSON-LD)
        """)

    # Debug information
    with st.expander("🔍 Debug Information"):
        st.markdown("**Content Area Word Counts:**")
        debug_data = []
        for area, data in content_areas.items():
            if data['content'].strip():
                word_count = len([word for word in data['content'].split() if word.strip()])
                debug_data.append({
                    'Area': area.replace('_', ' ').title(),
                    'Weight': f"{data['weight']}x",
                    'Words': word_count,
                    'Characters': len(data['content']),
                    'Preview': data['content'][:100] + '...' if len(data['content']) > 100 else data['content']
                })

        if debug_data:
            df = pd.DataFrame(debug_data)
            st.dataframe(df, use_container_width=True)

        st.markdown("**Analysis Details:**")
        debug_info = {
            'Soft 404 Analysis': soft_404_result,
            'Readability Data': readability_data,
            'Token Count': len(importance_scores),
            'JavaScript Content Length': len(content_areas['javascript']['content'])
        }
        st.json(debug_info)

        if importance_scores:
            st.markdown("**Top 10 Token Scores:**")
            top_10 = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10])
            st.json(top_10)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Google Content Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()
    download_nltk_data()
    create_app_header()

    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Options")
        input_method = st.radio(
            "Choose input method:",
            ["🌐 Analyze URL", "📝 Paste HTML", "🎯 Load Sample"],
            help="Select how you want to provide content"
        )

        st.markdown("---")
        st.markdown("### 📊 Display Options")
        show_advanced_charts = st.checkbox("Show Advanced Charts", value=True)

        st.markdown("---")
        st.markdown("""
        ### 📖 Gary Illyes' Method
        **Enhanced Features:**
        - 🎯 **Main Content**: 10x weight priority
        - ⚡ **JavaScript Analysis**: Extract structured data
        - 📍 **Position Weighting**: Location-based scoring
        - 🔤 **Smart Tokenization**: Enhanced filtering
        - ⚠️ **Soft 404 Detection**: Multi-factor analysis
        """)

    # Main content handling
    if input_method == "🌐 Analyze URL":
        url = st.text_input("🔗 Enter URL:", placeholder="https://example.com")
        if url and st.button("🚀 Analyze Website"):
            if url.startswith(('http://', 'https://')):
                analyze_url(url, show_advanced_charts)
            else:
                st.error("❌ Please enter a valid URL starting with http:// or https://")

    elif input_method == "📝 Paste HTML":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            html_content = st.text_area(
                "📝 Paste HTML content:",
                height=300,
                placeholder="<html><body>Your HTML content here...</body></html>",
                help="Paste the HTML source code to analyze"
            )
        
        with col2:
            st.markdown("**🎯 Quick Fill:**")
            if st.button("📝 E-commerce Page", key="html_ecommerce"):
                st.session_state.sample_html = SAMPLE_HTML_OPTIONS["E-commerce Product Page"]
                st.rerun()
            if st.button("📝 Blog Article", key="html_blog"):
                st.session_state.sample_html = SAMPLE_HTML_OPTIONS["Blog Article"]
                st.rerun()
            if st.button("📝 Soft 404 Page", key="html_404"):
                st.session_state.sample_html = SAMPLE_HTML_OPTIONS["Soft 404 Example"]
                st.rerun()
            if st.button("🧹 Clear", key="html_clear"):
                st.session_state.sample_html = ""
                st.rerun()
        
        # Check if sample HTML was selected
        if 'sample_html' in st.session_state and st.session_state.sample_html:
            html_content = st.session_state.sample_html
            st.success("✅ Sample HTML loaded! You can edit it below or analyze as-is.")
            
            # Show editable text area with sample content
            html_content = st.text_area(
                "📝 Edit HTML content (sample loaded):",
                value=html_content,
                height=300,
                help="Sample HTML loaded - you can edit it or analyze as-is"
            )
        
        if html_content and st.button("🔬 Analyze HTML"):
            analyze_html(html_content, "Pasted HTML", show_advanced_charts)

    else:  # Load Sample
        st.markdown('<div class="sample-selector">', unsafe_allow_html=True)
        st.markdown("### 🎯 Choose a Sample to Analyze")

        sample_choice = st.selectbox(
            "Select sample content:",
            list(SAMPLE_HTML_OPTIONS.keys()),
            help="Choose from pre-built samples to see how the analyzer works"
        )

        if st.button("📊 Analyze Sample"):
            html_content = SAMPLE_HTML_OPTIONS[sample_choice]
            analyze_html(html_content, f"Sample: {sample_choice}", show_advanced_charts)

        st.markdown('</div>', unsafe_allow_html=True)

        # Show sample preview
        if sample_choice:
            with st.expander(f"👁️ Preview: {sample_choice}"):
                st.code(SAMPLE_HTML_OPTIONS[sample_choice][:500] + "...", language="html")

if __name__ == "__main__":
    main() 
