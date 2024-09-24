import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        for review in reviews:
            if isinstance(review['Timestamp'], str):
                review['Timestamp'] = datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S')
        self.valid_locations = [
            "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
            "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
            "El Paso, Texas", "Escondido, California", "Fresno, California",
            "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
            "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
            "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
        ]

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query = parse_qs(environ['QUERY_STRING'])
            location = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]

            filtered_reviews = reviews

            if location:
                filtered_reviews = [r for r in filtered_reviews if r['Location'] == location]

            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [r for r in filtered_reviews if r['Timestamp'] >= start_date]

            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [r for r in filtered_reviews if r['Timestamp'] <= end_date]

            # Analyze sentiment for each review
            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = sentiment

            # Sort reviews by compound sentiment score in descending order
            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

            # Format the response
            response = []
            for review in sorted_reviews:
                response.append({
                    'ReviewId': review['ReviewId'],
                    'Location': review['Location'],
                    'Timestamp': review['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'ReviewBody': review['ReviewBody'],
                    'sentiment': review['sentiment']
                })

            response_body = json.dumps(response, indent=2).encode("utf-8")


            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        elif environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                post_params = parse_qs(post_data)

                review_body = post_params.get('ReviewBody', [''])[0]
                location = post_params.get('Location', [''])[0]

                if not review_body or not location:
                    raise ValueError('ReviewBody and Location are required')

                if location not in self.valid_locations:
                    raise ValueError('Invalid Location')

                new_review = {
                    'ReviewId': str(uuid.uuid4()),
                    'ReviewBody': review_body,
                    'Location': location,
                    'Timestamp': datetime.now()
                }

                reviews.append(new_review)

                response = {
                    'ReviewId': new_review['ReviewId'],
                    'ReviewBody': new_review['ReviewBody'],
                    'Location': new_review['Location'],
                    'Timestamp': new_review['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                }

                response_body = json.dumps(response, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]

            except Exception as e:
                error_message = str(e)
                response_body = json.dumps({"error": error_message}, indent=2).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

        else:
            response_body = json.dumps({"error": "Method not allowed"}, indent=2).encode("utf-8")
            start_response("405 Method Not Allowed", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()