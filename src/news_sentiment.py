"""
News Sentiment Analysis Module
Fetch and analyze news sentiment for stocks
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Analyze news sentiment for stocks"""
    
    def __init__(self, api_key=None):
        """
        Initialize news sentiment analyzer
        
        Args:
            api_key (str): API key for news service (optional for demo mode)
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch_news(self, ticker, days=7, max_articles=50):
        """
        Fetch news articles for a stock
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to look back
            max_articles (int): Maximum number of articles
            
        Returns:
            list: List of news articles
        """
        try:
            if not self.api_key:
                # Return demo data if no API key
                return self._get_demo_news(ticker, days)
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': f'{ticker} stock OR {ticker} shares',
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': max_articles,
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                logger.info(f"Fetched {len(articles)} news articles for {ticker}")
                return articles
            else:
                logger.warning(f"Failed to fetch news: {response.status_code}")
                return self._get_demo_news(ticker, days)
                
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return self._get_demo_news(ticker, days)
    
    def _get_demo_news(self, ticker, days=7):
        """Generate demo news articles"""
        demo_articles = [
            {
                'title': f'{ticker} Reports Strong Quarterly Earnings',
                'description': f'{ticker} exceeded analyst expectations with strong revenue growth and positive guidance for the next quarter.',
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat(),
                'source': {'name': 'Financial Times'},
                'url': '#'
            },
            {
                'title': f'Analysts Upgrade {ticker} Stock Rating',
                'description': f'Major investment bank upgrades {ticker} to "Buy" with increased price target.',
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat(),
                'source': {'name': 'Bloomberg'},
                'url': '#'
            },
            {
                'title': f'{ticker} Faces Market Headwinds',
                'description': f'Industry challenges and increased competition create uncertainty for {ticker}.',
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat(),
                'source': {'name': 'CNBC'},
                'url': '#'
            },
            {
                'title': f'{ticker} Announces New Product Launch',
                'description': f'{ticker} unveils innovative product line expected to drive future growth.',
                'publishedAt': (datetime.now() - timedelta(days=4)).isoformat(),
                'source': {'name': 'Reuters'},
                'url': '#'
            },
            {
                'title': f'Market Analysis: {ticker} Shows Resilience',
                'description': f'{ticker} stock demonstrates strength amid market volatility, attracting investor interest.',
                'publishedAt': (datetime.now() - timedelta(days=5)).isoformat(),
                'source': {'name': 'Wall Street Journal'},
                'url': '#'
            }
        ]
        return demo_articles[:min(len(demo_articles), days)]
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not text:
            return {'polarity': 0, 'subjectivity': 0, 'label': 'neutral'}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'label': label
        }
    
    def get_news_sentiment(self, ticker, days=7, max_articles=50):
        """
        Get comprehensive news sentiment analysis
        
        Args:
            ticker (str): Stock ticker
            days (int): Days to look back
            max_articles (int): Max articles to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        articles = self.fetch_news(ticker, days, max_articles)
        
        if not articles:
            return {
                'articles': [],
                'summary': {
                    'total_articles': 0,
                    'avg_sentiment': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'overall_label': 'neutral'
                }
            }
        
        analyzed_articles = []
        sentiments = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            
            analyzed_articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published': article.get('publishedAt', ''),
                'url': article.get('url', '#'),
                'sentiment': sentiment
            })
            
            sentiments.append(sentiment['polarity'])
        
        # Calculate summary statistics
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Overall label
        if avg_sentiment > 0.1:
            overall_label = 'positive'
        elif avg_sentiment < -0.1:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        return {
            'articles': analyzed_articles,
            'summary': {
                'total_articles': len(articles),
                'avg_sentiment': avg_sentiment,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_pct': (positive_count / len(sentiments) * 100) if sentiments else 0,
                'negative_pct': (negative_count / len(sentiments) * 100) if sentiments else 0,
                'neutral_pct': (neutral_count / len(sentiments) * 100) if sentiments else 0,
                'overall_label': overall_label
            }
        }
    
    @staticmethod
    def get_sentiment_color(label):
        """Get color for sentiment label"""
        colors = {
            'positive': '#10b981',
            'negative': '#ef4444',
            'neutral': '#f59e0b'
        }
        return colors.get(label, '#6b7280')
    
    @staticmethod
    def get_sentiment_emoji(label):
        """Get emoji for sentiment label"""
        emojis = {
            'positive': '😊',
            'negative': '😟',
            'neutral': '😐'
        }
        return emojis.get(label, '➖')
