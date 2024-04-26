from flask import jsonify, request
from services.sitemap_scraper import scrape_sitemap
from services.embeddings import manage_embeddings
from services.text_processing import load_and_split_text

def setup_routes(app):
    @app.route('/processSitemap', methods=['POST'])
    def process_sitemap():
        sitemap_url = request.json['sitemap_url']
        urls = scrape_sitemap(sitemap_url).split(', ')
        return jsonify(status="Sitemap processed", urls=urls)

    @app.route('/askQuestion', methods=['POST'])
    def ask_question():
        question = request.json['question']
        # Handle the question using a model or logic
        answer = "Sample Answer to " + question  # Placeholder for actual processing
        return jsonify(response=answer, status="Success")
